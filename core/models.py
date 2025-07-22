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
        self.embed_sz = embedding_size
        self.n_heads = number_of_heads
        self.head_dims = embedding_size // number_of_heads

        # - Make the Values, Keys and Queries matrices
        self.values = torch.nn.Linear(self.head_dims, self.head_dims, bias=False)
        self.keys = torch.nn.Linear(self.head_dims, self.head_dims, bias=False)
        self.queries = torch.nn.Linear(self.head_dims, self.head_dims, bias=False)

        # - Output layer
        self.fc_out = torch.nn.Linear(self.n_heads * self.head_dims, self.embed_sz)

    def forward(self, values, keys, queries, mask: torch.Tensor):
        N = queries.shape[0]
        val_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # - Split embedding into self.heads pieces
        values = values.reshape(N, val_len, self.n_heads, self.head_dims)
        keys = keys.reshape(N, key_len, self.n_heads, self.head_dims)
        queries = queries.reshape(N, query_len, self.n_heads, self.head_dims)

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

        attention = torch.softmax(energy / (self.embed_sz ** (1/2)), dim=3)

        # - The shapes are:
        # attention: (N, heads, query_len, key_len)
        # values: (N, val_len, heads, heads_dim) {key_len == val_len}
        # out: (N, query_len, heads, head_dim)
        out = torch.einsum('nhql, nlhd -> nqhd', [attention, values]).reshape(N, query_len, self.n_heads * self.head_dims)

        # - Run it through the out layer
        out = self.fc_out(out)

        return out


class TransformerBlock(torch.nn.Module):
    def __init__(self, embedding_size, number_of_heads, p_dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embedding_size=embedding_size, number_of_heads=number_of_heads)
        self.norm1 = torch.nn.LayerNorm(embedding_size)  # normalizes per example instead of batch
        self.norm2 = torch.nn.LayerNorm(embedding_size)

        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, forward_expansion * embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(forward_expansion * embedding_size, embedding_size)
        )

        self.dropout = torch.nn.Dropout(p_dropout)

    def forward(self, values, keys, queries, mask):
        attention = self.attention(values, keys, queries, mask)

        X = self.dropout(self.norm1(attention + queries))

        fwrd = self.feed_forward(X)

        out = self.dropout(self.norm2(fwrd + X))

        return out


class Encoder(torch.nn.Module):
    def __init__(self, source_vocabulary_size, embedding_size, number_of_layers, number_of_heads, forward_expansion, p_dropout, max_length, device):
        super().__init__()
        self.embed_size = embedding_size

        self.device = device
        self.word_embedding = torch.nn.Embedding(source_vocabulary_size, embedding_size)
        self.position_embedding = torch.nn.Embedding(350, embedding_size)
        self.dropout = torch.nn.Dropout(p_dropout)

        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(embedding_size=embedding_size, number_of_heads=number_of_heads, p_dropout=p_dropout, forward_expansion=forward_expansion)
            for _ in range(number_of_layers)]
        )

    def forward(self, X, mask):
        N, seq_len = X.shape
        pos = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        # out = self.dropout(X + pos)
        # out = self.dropout(X + self.position_embedding(pos))
        out = self.dropout(self.word_embedding(X) + self.position_embedding(pos))

        for lyr in self.layers:
            out = lyr(out, out, out, mask)

        return out


class DecoderBlock(torch.nn.Module):
    def __init__(self, embedding_size, number_of_heads, forward_expansion, p_dropout, device):
        super().__init__()
        self.attention = SelfAttention(embedding_size=embedding_size, number_of_heads=number_of_heads)
        self.norm_lyr = torch.nn.LayerNorm(embedding_size)
        self.trans_blk = TransformerBlock(
            embedding_size=embedding_size,
            number_of_heads=number_of_heads,
            p_dropout=p_dropout,
            forward_expansion=forward_expansion
        )

        self.dropout = torch.nn.Dropout(p_dropout)

    def forward(self, X, values, keys, source_mask, target_mask):
        att = self.attention(X, X, X, target_mask)
        qrs = self.dropout(self.norm_lyr(att + X))
        out = self.trans_blk(values, keys, qrs, source_mask)
        return out


class Decoder(torch.nn.Module):
    def __init__(self, target_vocabulary_size, embedding_size, number_of_layers, number_of_heads, forward_expansion, p_dropout, max_length, device):
        super().__init__()
        self.device = device
        self.wrd_embed = torch.nn.Embedding(target_vocabulary_size, embedding_size)
        self.pos_embed = torch.nn.Embedding(max_length, embedding_size)

        self.lyrs = torch.nn.ModuleList([
            DecoderBlock(embedding_size=embedding_size, number_of_heads=number_of_heads, forward_expansion=forward_expansion, p_dropout=p_dropout, device=device)
        ])

        self.fc_out = torch.nn.Linear(embedding_size, target_vocabulary_size)

        self.dropout = torch.nn.Dropout(p_dropout)

    def forward(self, X, enc_out, src_mask, trg_mask):
        N, seq_length = X.shape
        pos = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        X = self.dropout((self.wrd_embed(X) + self.pos_embed(pos)))

        for lyr in self.lyrs:
            X = lyr(X, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(X)

        return out


def get_source_mask(source: torch.Tensor, source_padding_index: int):
    # - The shape should be:
    # (batch_size, 1, 1, src_len)
    src_msk = (source != source_padding_index).unsqueeze(1).unsqueeze(2)

    return src_msk


def get_target_mask(target: torch.Tensor, target_padding_index: int):
    btch_sz, trg_len = target.shape
    trg_msk = torch.tril(torch.ones((trg_len, trg_len))).expand(btch_sz, 1, trg_len, trg_len)

    return trg_msk


class Transformer(torch.nn.Module):
    def __init__(self, source_vocabulary_size, target_vocabulary_size, source_padding_index, target_padding_index, embedding_size, number_of_layers, number_of_heads, forward_expansion, p_dropout, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), max_length=100):
        super().__init__()

        self.enc = Encoder(source_vocabulary_size=source_vocabulary_size, embedding_size=embedding_size, number_of_layers=number_of_layers, number_of_heads=number_of_heads, device=device, forward_expansion=forward_expansion, p_dropout=p_dropout, max_length=max_length)

        self.dec = Decoder(target_vocabulary_size=target_vocabulary_size, embedding_size=embedding_size, number_of_layers=number_of_layers, number_of_heads=number_of_heads, forward_expansion=forward_expansion, p_dropout=p_dropout, device=device, max_length=max_length)

        self.src_pad_idx = int(source_padding_index)
        self.trg_pad_idx = int(target_padding_index)
        self.device = device


    def forward(self, source, target):
        src_msk = get_source_mask(source=source, source_padding_index=self.src_pad_idx)
        trg_msk = get_target_mask(target=target, target_padding_index=self.trg_pad_idx)
        enc_src = self.enc(source, src_msk)
        out = self.dec(target, enc_src, src_msk, trg_msk)

        return out



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
    def __init__(self, head_model, in_channels: int,  out_channels: int):
        super().__init__()

        self.mdl = head_model
        self.in_chnls = in_channels
        self.out_chnls = out_channels

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
            self.out_chnls
        )
        print(f'''
- Model:
{self.mdl}
''')

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

    def __init__(self, model_name: str, n_features, code_length, layer_activation, reverse: bool = False, save_dir: str or pathlib.Path = pathlib.Path(
        '../output')):
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


class EncAttentionNet(torch.nn.Module):
    def __init__(self, head_model, embedding_size, number_of_heads):
        super().__init__()

        self.mdl = head_model
        self.embd_sz = int(embedding_size)
        self.n_heads = int(number_of_heads)
        self.att_lyr = SelfAttention(embedding_size=self.embd_sz, number_of_heads=self.n_heads)
        self.embd_lyr = torch.nn.Embedding(self.embd_sz, self.embd_sz)

    def forward(self, X):
        msk = get_target_mask(target=X, target_padding_index=0)
        X = self.embd_lyr(X)
        X = self.att_lyr(X, X, X, msk)

        b, h, w = X.shape

        X = self.mdl(X.view(b, 1, h, w), p_drop=0.0)

        return X


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    SRC_PAD_IDX = 0
    TRG_PAD_IDX = 0
    SRC_VOCAB_SIZE = 350
    TRG_VOCAB_SIZE = 350
    EMBEDDING_SIZE = 350
    NUMBER_OF_HEADS = 35
    NUMBER_OF_LAYERS = 6
    P_DROPOUT = 0
    FORWARD_EXPANSION = 4

    MODEL = torchvision.models.resnet18
    WEIGHTS = torchvision.models.ResNet18_Weights.IMAGENET1K_V1

    inputs = torch.tensor(np.random.randint(35, 350, (32, 350))).to(DEVICE)
    label = torch.tensor(np.random.randint(35, 350, (32, 350))).to(DEVICE)

    mdl = EncAttentionNet(
        head_model=EncResNet(
            head_model=MODEL(weights=WEIGHTS),
            in_channels=1,
            out_channels=1
        ),
        embedding_size=350,
        number_of_heads=35
    )

    out = mdl(inputs)

    print(out)

#     model = Transformer(
#     SRC_VOCAB_SIZE,
#     TRG_VOCAB_SIZE,
#     SRC_PAD_IDX,
#     TRG_PAD_IDX,
#     embedding_size=EMBEDDING_SIZE,
#     number_of_layers=NUMBER_OF_LAYERS,
#     number_of_heads=NUMBER_OF_HEADS,
#     p_dropout=P_DROPOUT,
#     forward_expansion=FORWARD_EXPANSION,
#     device=DEVICE
# ).to(DEVICE)
#

