# import torch
# from torchvision.models import resnet18
# import torch.nn.utils.prune as prune
# from torchsummary import summary
# 
# # =============================================================
# # NOTE: Begin Pruning Classifier
# # =============================================================
# dir_model = ".\\models_old\\"
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')
# 
# # model_name = "nabernet_a"
# # model = torch.load(dir_model + model_name + ".tar")
# model = resnet18(pretrained=False)
# model_name = "resnet18"
# 
# # model.eval()
# model.to(device)
# 
# IM_SIZE = 224
# summary(model, input_size=(3, IM_SIZE, IM_SIZE))
# 
# # prune_params = (
# #     (model.conv1, 'weight'),
# #     (model.conv2, 'weight'),
# #     (model.conv3, 'weight'),
# #     (model.conv4, 'weight'),
# #     (model.conv5, 'weight'),
# #     (model.fc1, 'weight'),
# #     (model.fc2, 'weight'),
# #     (model.fc3, 'weight'),
# # )
# 
# # prune.global_unstructured(
# #     prune_params,
# #     pruning_method=prune.L1Unstructured,
# #     amount=0.4,
# # )
# 
# pair = ('weight', 'bias')
# amount = 0.4
# 
# for name in pair:
#     prune.l1_unstructured(model.conv1, name, amount=amount)
#     prune.l1_unstructured(model.conv2, name, amount=amount)
#     prune.l1_unstructured(model.conv3, name, amount=amount)
#     prune.l1_unstructured(model.conv4, name, amount=amount)
#     prune.l1_unstructured(model.conv5, name, amount=amount)
#     prune.l1_unstructured(model.fc1, name, amount=amount)
#     prune.l1_unstructured(model.fc2, name, amount=amount)
#     prune.l1_unstructured(model.fc3, name, amount=amount)
# 
# with open("params1.txt", 'w') as f:
#     terms = list(model.named_parameters())
#     f.writelines(''.join(str(terms)))
# f.close()
# 
# with open("buffers1.txt", 'w') as f:
#     f.writelines(''.join(str(list(model.named_buffers()))))
# f.close()
# 
# for name in pair:
#     prune.remove(model.conv1, name)
#     prune.remove(model.conv2, name)
#     prune.remove(model.conv3, name)
#     prune.remove(model.conv4, name)
#     prune.remove(model.conv5, name)
#     prune.remove(model.fc1, name)
#     prune.remove(model.fc2, name)
#     prune.remove(model.fc3, name)
# 
# print("After Pruning:")
# with open("params2.txt", 'w') as f:
#     terms = list(model.named_parameters())
#     f.writelines(''.join(str(terms)))
# f.close()
# 
# summary(model, input_size=(3, IM_SIZE, IM_SIZE))
# # torch.save(model, dir_model + model_name + ".hdf5")
