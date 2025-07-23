import torch
from mdt.models.domain_adapt.wgan import Discriminator1d, Discriminator2d
from mdt.models.domain_adapt.wgan_simple import Discriminator1dImproved, Discriminator2dImproved
from mdt.models.perceptual_encoders.voltron_encoder import VoltronTokenEncoder

vis_feat = torch.randn(8, 1152).cuda()
net = Discriminator1dImproved(
    in_dim=1152,
    inner_dim=256,
    dropout=0.1,
).cuda()
print(net.calc_params())
net.debug = True
output = net(vis_feat)
print(output.shape)


act_feat = torch.randn(8, 10, 384).cuda()
net2 = Discriminator2dImproved(
    in_dim=384,
    inner_dim=256,
    condition_dim=384,
).cuda()
print(net2.calc_params())
output = net2(act_feat)
print(output.shape)


# voltron_encoder = VoltronTokenEncoder(
#     latent_dim=384,
#     model_type='v-cond',
#     device="cuda",
#     cache="/home/geyuan//pretrained",
# )
# # print(voltron_encoder.state_dict().keys())
# # print(voltron_encoder.vcond.encoder_blocks[-1].mlp[-1])
# voltron_encoder.freeze_backbone()
