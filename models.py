import pathlib
import numpy as np
import torch
import torchvision.models
from torch_geometric.nn import GCNConv
from transformers import AutoConfig, AutoModel
from sklearn.neighbors import KNeighborsClassifier


class SelfAttention(torch.nn.Module):
    def __init__(self, embedding_size: int, number_of_heads: int):
        super().__init__()
        assert (embedding_size % number_of_heads == 0), "embedding_size needs to be divisible by number_of_heads"
        self.embedding_size = embedding_size
        self.heads = number_of_heads
        self.head_dim = embedding_size // number_of_heads

        # - Make the Values, Keys and Queries matrices
        self.values = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)

        # - Output layer
        self.fc_out = torch.nn.Linear(number_of_heads * self.head_dim, embedding_size)

    def forward(self, values, keys, queries, mask: torch.Tensor):
        N = queries.shape[0]
        val_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # - Split embedding into self.heads pieces
        values = values.reshape(N, val_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # - Send through the linear layers
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # - The shapes are as follows:
        # queries: (N, query_len, heads, heads_dim)
        # keys: (N, key_len, heads, heads_dim)
        # emergy: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        attention = torch.softmax(energy / (self.embedding_size ** (1/2)), dim=3)

        # - The shapes are:
        # attention: (N, heads, query_len, key_len)
        # values: (N, val_len, heads, heads_dim) {key_len == val_len}
        # out: (N, query_len, heads, head_dim)
        out = torch.einsum('nhql, nlhd -> nqhd', [attention, values]).reshape(N, query_len, self.heads * self.head_dim)

        # - Run it through the out layer
        out = self.fc_out(out)

        return out


class TransformerBlock(torch.nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embedding_size=embed_size, number_of_heads=heads)
        self.norm1 = torch.nn.LayerNorm(embed_size)  # normalizes per example instead of batch
        self.norm2 = torch.nn.LayerNorm(embed_size)

        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embed_size, forward_expansion * embed_size),
            torch.nn.ReLU(),
            torch.nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, values, keys, queries, mask):
        attention = self.attention(values, keys, queries, mask)

        X = self.dropout(self.norm1(attention + queries))

        fwrd = self.feed_forward(X)

        out = self.dropout(self.norm2(fwrd + X))

        return out


class Encoder(torch.nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super().__init__()
        self.embed_size = embed_size

        self.device = device
        self.word_embedding = torch.nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = torch.nn.Embedding(max_length, embed_size)
        self.dropout = torch.nn.Dropout(dropout)

        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(embed_size=embed_size, heads=heads, dropout=dropout, forward_expansion=forward_expansion)
            for _ in range(num_layers)]
        )

    def forward(self, X, mask):
        N, seq_len = X.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        out = self.dropout(self.word_embedding(X) + self.position_embedding(positions))

        for lyr in self.layers:
            out = lyr(out, out, out, mask)

        return out


class DecoderBlock(torch.nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super().__init__()
        self.attention = SelfAttention(embedding_size=embed_size, number_of_heads=heads)
        self.norm = torch.nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size=embed_size, heads=heads, dropout=dropout, forward_expansion=forward_expansion
        )

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, X, values, keys, src_mask, trg_mask):
        attention = self.attention(X, X, X, trg_mask)
        queries = self.dropout(self.norm(attention + X))
        out = self.transformer_block(values, keys, queries, src_mask)
        return out


class Decoder(torch.nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super().__init__()
        self.device = device
        self.word_embedding = torch.nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = torch.nn.Embedding(max_length, embed_size)

        self.layers = torch.nn.ModuleList([
            DecoderBlock(embed_size=embed_size, heads=heads, forward_expansion=forward_expansion, dropout=dropout, device=device)
        ])

        self.fc_out = torch.nn.Linear(embed_size, trg_vocab_size)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, X, enc_out, src_mask, trg_mask):
        N, seq_length = X.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        X = self.dropout((self.word_embedding(X) + self.position_embedding(positions)))

        for lyr in self.layers:
            X = lyr(X, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(X)

        return out


class Transformer(torch.nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=256, num_layers=6, forward_expansion=4, heads=8, dropout=0, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), max_length=100):
        super().__init__()

        self.enc = Encoder(src_vocab_size=src_vocab_size, embed_size=embed_size, num_layers=num_layers, heads=heads, device=device, forward_expansion=forward_expansion, dropout=dropout, max_length=max_length)

        self.dec = Decoder(trg_vocab_size=trg_vocab_size, embed_size=embed_size, num_layers=num_layers, heads=heads, forward_expansion=forward_expansion, dropout=dropout, device=device, max_length=max_length)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src: torch.Tensor):
        # - The shape should be:
        # (N, 1, 1, src_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask

    def make_trg_mask(self, trg: torch.Tensor):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src=src)
        trg_mask = self.make_trg_mask(trg=trg)
        enc_src = self.enc(src, src_mask)
        out = self.dec(trg, enc_src, src_mask, trg_mask)

        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)

    out = model(x, trg[:, :-1])
    print(out)


class EncEfficientNetV2(torch.nn.Module):
    def __init__(self, architecture: str, in_channels: int,  out_size: int, pretrained: bool = False, freeze_layers: bool = False):
        super().__init__()

        self.base_mdl = None
        if architecture == 's':
            self.base_mdl = torchvision.models.efficientnet_v2_s
            self.weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        elif architecture == 'm':
            self.base_mdl = torchvision.models.efficientnet_v2_m
            self.weights = torchvision.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
        elif architecture == 'l':
            self.base_mdl = torchvision.models.efficientnet_v2_l
            self.weights = torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
        self.in_chnls = in_channels
        self.out_sz = out_size
        self.pretrained = pretrained
        self.freeze_layers = freeze_layers

        if self.base_mdl is not None:
            self.make_model()

    def make_model(self):
        # - Load the weights
        if self.pretrained:
            self.base_mdl = self.base_mdl(weights=self.weights)
        else:
            self.base_mdl = self.base_mdl()


        # - Freeze all the layers
        if self.freeze_layers:
            for param in self.base_mdl.parameters():
                param.requires_grad = False

        # - Modify the stem (first conv) to handle 5x5 input
        # Default: Conv2d(3, 24, kernel_size=3, stride=2, padding=1)
        # Change stride 3->1 so that 5x5 input doesn't collapse
        new_conv = torch.nn.Conv2d(
            in_channels=self.in_chnls, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False
        )
        torch.nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        self.base_mdl.features[0][0] = new_conv

        # - Replace the head (predictor layer) with regression
        in_feats = self.base_mdl.classifier[1].in_features
        self.base_mdl.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_feats, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x, p_drop):
        x = self.base_mdl(x)
        x = torch.nn.functional.dropout(x, p=p_drop, training=self.training)
        return x


class EncResNet(torch.nn.Module):
    def __init__(self, head_model, in_channels: int,  out_size: int):
        super().__init__()

        self.mdl = head_model
        self.in_chnls = in_channels
        self.out_sz = out_size

        self.make_model()

    def make_model(self):
        fst_lyr = self.mdl.conv1
        self.mdl.conv1 = torch.nn.Conv2d(
            self.in_chnls,
            fst_lyr.out_channels,
            kernel_size=(fst_lyr.kernel_size[0], fst_lyr.kernel_size[1]),
            stride=(fst_lyr.stride[0], fst_lyr.stride[1]),
            padding=fst_lyr.padding,
            bias=False if fst_lyr.bias is None else fst_lyr.bias
        )

        lst_lyr = self.mdl.fc
        self.mdl.fc = torch.nn.Linear(
            lst_lyr.in_features,
            self.out_sz
        )

    def forward(self, x, p_drop):
        x = self.mdl(x)
        x = torch.nn.functional.dropout(x, p=p_drop, training=self.training)
        return x


class AutoEncoder(torch.nn.Module):
    class Encoder(torch.nn.Module):
        def __init__(self, in_units, code_length, layer_activation):
            super().__init__()
            self._in_units = in_units
            self._code_length = code_length
            self.layers = torch.nn.ModuleList()
            self.layer_activation = layer_activation
            self._build()

        def _build(self):
            in_units = self._in_units
            out_units = in_units

            # - Encoder
            while out_units > self._code_length:
                # - Add a layer
                self._add_layer(
                    n_in=in_units,
                    n_out=out_units // 2,
                )
                out_units //= 2
                in_units = out_units

        def _add_layer(self, n_in, n_out):
            self.layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(n_in, n_out),
                    torch.nn.BatchNorm1d(n_out),
                    self.layer_activation()
                )
            )

        def forward(self, x, max_p_drop: float = 0.0):
            for lyr in self.layers:
                x = lyr(x)
                p_drop = np.random.random() * max_p_drop
                x = torch.nn.functional.dropout(x, p=p_drop, training=self.training)
            return x

        def save_weights(self, filename: str or pathlib.Path, verbose: bool = False):
            if verbose:
                print(f'=> (ENCODER) Saving checkpoint to \'{filename}\' ...')
            torch.save(self.state_dict(), filename)

        def load_weights(self, weights_file, verbose: bool = False):
            if verbose:
                print('=> (ENCODER) Loading checkpoint ...')
            self.load_state_dict(torch.load(weights_file, weights_only=True))

        @property
        def code_length(self):
            return self._code_length

        @code_length.setter
        def code_length(self, value):
            self._code_length = value

        @property
        def in_units(self):
            return self._in_units

        @in_units.setter
        def in_units(self, value):
            self._in_units = value

    class Decoder(torch.nn.Module):
        def __init__(self, code_length, out_units, layer_activation):
            super().__init__()
            self._code_length = code_length
            self._out_units = out_units
            self.layer_activation = layer_activation
            self.layers = torch.nn.ModuleList()
            self._build()

        def _build(self):
            code_length = self._code_length
            in_units = code_length

            while code_length < self._out_units:
                # - Add a layer
                self._add_layer(
                    n_in=in_units,
                    n_out=code_length * 2,
                )
                code_length *= 2
                in_units = code_length

        def _add_layer(self, n_in, n_out):
            self.layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(n_in, n_out),
                    torch.nn.BatchNorm1d(n_out),
                    self.layer_activation()
                )
            )

        def forward(self, x, max_p_drop: float = 0.0):
            for lyr in self.layers:
                x = lyr(x)
                p_drop = np.random.random() * max_p_drop
                x = torch.nn.functional.dropout(x, p=p_drop, training=self.training)
            return x

        def save_weights(self, filename: str or pathlib.Path, verbose: bool = False):
            if verbose:
                print(f'=> (DECODER) Saving weights to \'{filename}\' ...')
            torch.save(self.state_dict(), filename)

        def load_weights(self, weights_file, verbose: bool = False):
            if verbose:
                print('=> (DECODER) Loading weights ...')
            self.load_state_dict(torch.load(weights_file, weights_only=True))

        @property
        def out_units(self):
            return self._out_units

        @out_units.setter
        def out_units(self, value):
            self._out_units = value

        @property
        def code_length(self):
            return self._code_length

        @code_length.setter
        def code_length(self, value):
            self._code_length = value

    def __init__(self, model_name: str, n_features, code_length, layer_activation, reverse: bool = False, save_dir: str or pathlib.Path = pathlib.Path('./output')):
        super().__init__()
        self.model_name = model_name
        self.n_features = n_features
        self.code_length = code_length
        self.layer_activation = layer_activation
        self.reverse = reverse
        self.encoder = None
        self.decoder = None

        # - Make sure the save_dir exists and of type pathlib.Path
        assert isinstance(save_dir, str) or isinstance(save_dir, pathlib.Path), f'AE: save_dir must be of type str or pathlib.Path, but is of type {type(save_dir)}!'
        self.save_dir = save_dir
        # os.makedirs(self.save_dir, exist_ok=True)
        self.save_dir = pathlib.Path(self.save_dir)

        self._build()

    def _build(self):
        if self.reverse:
            self.decoder = self.Decoder(
                code_length=self.n_features,
                out_units=self.code_length,
                layer_activation=self.layer_activation
            )
            self.encoder = self.Encoder(
                in_units=self.decoder.out_units,
                code_length=self.n_features,
                layer_activation=self.layer_activation
            )
        else:
            self.encoder = self.Encoder(
                in_units=self.n_features,
                code_length=self.code_length,
                layer_activation=self.layer_activation
            )
            self.decoder = self.Decoder(
                code_length=self.encoder.code_length,
                out_units=self.n_features,
                layer_activation=self.layer_activation
            )

    def forward(self, x, p_drop: float = 0.0):
        if self.reverse:
            x = self.decoder(x, max_p_drop=p_drop)
            x = self.encoder(x, max_p_drop=p_drop)
        else:
            x = self.encoder(x, max_p_drop=p_drop)
            x = self.decoder(x, max_p_drop=p_drop)

        # - Save the weights of the encoder and the decoder separately
        self.encoder.save_weights(filename=self.save_dir / f'encoder_weights.pth')
        self.decoder.save_weights(filename=self.save_dir / f'decoder_weights.pth')

        return x


class QoENet1D(torch.nn.Module):
    def __init__(self, model_name: str, input_size, output_size, n_units, n_layers):
        super().__init__()
        self.model_name = model_name
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.n_units = n_units
        self.model = None
        self.layers = torch.nn.ModuleList()
        self._build()

    def _add_layer(self, n_in, n_out, activation):
        self.layers.append(
            torch.nn.Sequential(
                torch.nn.Linear(n_in, n_out),
                torch.nn.BatchNorm1d(n_out),
                activation()
            )
        )

    def _build(self):
        # - Add the input layer
        self._add_layer(n_in=self.input_size, n_out=self.n_units, activation=torch.nn.SiLU)

        for lyr in range(self.n_layers):
            self._add_layer(n_in=self.n_units, n_out=self.n_units, activation=torch.nn.SiLU)

        self._add_layer(n_in=self.n_units, n_out=self.output_size, activation=torch.nn.ReLU)

    def forward(self, x, p_drop: float = 0.0):
        tmp_in = x
        for lyr in self.layers:

            x = lyr(x)

            # - Skip connection
            if tmp_in.shape == x.shape:
                x = x + tmp_in

            tmp_in = x

        x = torch.nn.functional.dropout(x, p=p_drop, training=self.training)

        return x


class GCNClassifier(torch.nn.Module):
    def __init__(self, model_name: str, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.model_name = model_name
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)

        return x


class GRAGDataset(torch.utils.data.Dataset):
    def __init__(self, knn_neighbors: int = 5):
        super().__init__()
        self.knn_classifier = KNeighborsClassifier(n_neighbors=knn_neighbors)
        self.edges = None

    def get_edges(self, X, y):
        self.edges = self.knn_classifier.fit(X, y)


class GCNRegressor(torch.nn.Module):
    def __init__(self, model_name: str, in_channels: int, hidden_channels: int):
        super().__init__()
        self.model_name = model_name
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x.squeeze()


class TransformerForRegression(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.transformer = AutoModel.from_pretrained(self.model_name, config=self.config)
        self.regressor = torch.nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.regressor(pooled_output)
