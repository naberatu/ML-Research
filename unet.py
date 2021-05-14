# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms
#
#
# class Block(nn.Module):
#     def __init__(self, inc, outc):
#         super().__init__()
#         self.conv1 = nn.Conv2d(inc, outc, 3)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(outc, outc, 3)
#
#     def forward(self, x):
#         return self.relu(self.conv2(self.relu(self.conv1(x))))
#
#
# class Encoder(nn.Module):
#     def __init__(self, dims=(3, 64, 128, 256, 512, 1024)):
#         super().__init__()
#         self.encoder_blocks = nn.ModuleList([Block(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
#         self.pool = nn.MaxPool2d(2)
#
#     def forward(self, x):
#         features = []
#         for block in self.encoder_blocks:
#             x = block(x)
#             features.append(x)
#             x = self.pool(x)
#         return features
#
#
# class Decoder(nn.Module):
#     def __init__(self, dims=(1024, 512, 256, 128, 64)):
#         super().__init__()
#         self.dims = dims
#         self.upconvs = nn.ModuleList([nn.ConvTranspose2d(dims[i], dims[i + 1], 2, 2) for i in range(len(dims)-1)])
#         self.decoder_blocks = nn.ModuleList([Block(dims[i], dims[i + 1]) for i in range(len(dims)-1)])
#
#     def forward(self, x, encoder_features):
#         for i in range(len(self.dims) - 1):
#             x = self.upconvs[i](x)
#             enc_feats = self.crop(encoder_features[i], x)
#             x = torch.cat([x, enc_feats], dim=1)
#             x = self.decoder_blocks[i](x)
#
#     def crop(self, enc_feats, x):
#         _, _, H, W = x.shape
#         enc_feats = torchvision.transforms.CenterCrop([H, W])(enc_feats)
#         return enc_feats
#
#
# class UNet(nn.Module):
#     def __init__(self, enc_dims=(3, 64, 128, 256, 512, 1024), dec_dims=(1024, 512, 256, 128, 64), num_classes=1,
#                  retain_dim=False, out_size=(572, 572)):
#         super().__init__()
#         self.encoder = Encoder(enc_dims)
#         self.decoder = Decoder(dec_dims)
#         self.head = nn.Conv2d(dec_dims[-1], num_classes, 1)
#         self.retain_dim = retain_dim
#         self.out_size = out_size
#
#     def forward(self, x):
#         enc_features = self.encoder(x)
#         out = self.decoder(enc_features[::-1][0], enc_features[::-1][1:])
#         out = self.head(out)
#         if self.retain_dim:
#             out = F.interpolate(out, self.out_size)
#         return out
#
#
