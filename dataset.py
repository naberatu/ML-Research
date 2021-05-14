
import os
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix


# HELPER FUNCTION
def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data


# =============================================================
# NOTE DATASET CLASS
# =============================================================
class CTDataset(Dataset):
    def __init__(self, root_dir, classes="", covid_files="", non_covid_files="", transform=None, is_ctx=False, is_nii=False):
        self.root_dir = root_dir
        self.classes = classes
        self.files_path = [non_covid_files, covid_files]
        self.image_list = []
        self.transform = transform
        self.is_nii = is_nii


        # if not self.is_nii:
        # read the files from data split text files
        covid_files = read_txt(covid_files)
        non_covid_files = read_txt(non_covid_files)

        # combine the positive and negative files into a cumulative files list
        # if self.is_nii:
        #     self.image_list = nib.load(self.root_dir)
        #     # print(self.image_list.get_data().shape)
        #     img = np.array(self.image_list.dataobj)
        #     img = Image.fromarray(img.astype('uint8'), 'RGB')
        #     self.image_list = self.transform(img)
        #
        #     # plt.imshow(img[0])
        #     # plt.show()
        #     # exit(0)
        # else:
        for cls_index in range(len(self.classes)):
            if is_ctx:
                class_files = [[os.path.join(self.root_dir, x), cls_index] for x in read_txt(self.files_path[cls_index])]
            else:
                class_files = [[os.path.join(self.root_dir, self.classes[cls_index], x), cls_index] for x in read_txt(
                    self.files_path[cls_index])]
            self.image_list += class_files

    def __len__(self):
        # if self.is_nii:
        #     return len(self.image_list.dataobj)
        # else:
        return len(self.image_list)

    def __getitem__(self, idx):
        # if self.is_nii:
        #     nii_img = self.image_list[idx]
        #     data = torch.from_numpy(np.asarray(nii_img.dataobj))
        #
        #     img = Image.open(nii_img).convert('RGB')
        #     if self.transform:
        #         img = self.transform(img)
        #
        #     plt.imshow(img)
        #     plt.show()
        #     exit()
        #
        #     target = 'a'
        #     return data, target
        # else:
        path = self.image_list[idx][0]

        # Read the image
        image = Image.open(path).convert('RGB')     # For jpeg images

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = int(self.image_list[idx][1])

        data = {'img': image,
                'label': label}
                # 'paths': path}      # NOTE: Comment this line & compute_metrics out for fit_routine

        return data


# =============================================================
# Former Metrics Method (For Old_Main)
# =============================================================
# def compute_metrics(model, test_loader, device, plot_roc_curve=False):
#     model.eval()
#
#     val_loss = 0
#     val_correct = 0
#
#     criterion = nn.CrossEntropyLoss()
#
#     score_list = torch.Tensor([]).to(device)
#     pred_list = torch.Tensor([]).to(device).long()
#     target_list = torch.Tensor([]).to(device).long()
#     path_list = []
#
#     for iter_num, data in enumerate(test_loader):
#         # Convert image data into single channel data
#         image, target = data['img'].to(device), data['label'].to(device)
#         paths = data['paths']
#         path_list.extend(paths)
#
#         # Compute the loss
#         with torch.no_grad():
#             output = model(image)
#
#         # Log loss
#         val_loss += criterion(output, target.long()).item()
#
#         # Calculate the number of correctly classified examples
#         pred = output.argmax(dim=1, keepdim=True)
#         val_correct += pred.eq(target.long().view_as(pred)).sum().item()
#
#         # Bookkeeping
#         # score_list = torch.cat([score_list, nn.Softmax(dim=1)(output)[:, 1].squeeze()])
#         score_list = torch.cat([score_list, nn.Softmax(dim=1)(output)[:, 0].squeeze()])
#         pred_list = torch.cat([pred_list, pred.squeeze()])
#         target_list = torch.cat([target_list, target.squeeze()])
#
#     classification_metrics = classification_report(target_list.tolist(), pred_list.tolist(),
#                                                    target_names=['CT_NonCOVID', 'CT_COVID'],
#                                                    # target_names=['CT_COVID'],
#                                                    output_dict=True)
#
#     # sensitivity is the recall of the positive class
#     sensitivity = classification_metrics['CT_COVID']['recall']
#
#     # specificity is the recall of the negative class
#     specificity = classification_metrics['CT_NonCOVID']['recall']
#
#     # accuracy
#     accuracy = classification_metrics['accuracy']
#
#     # confusion matrix
#     conf_matrix = confusion_matrix(target_list.tolist(), pred_list.tolist())
#
#     # roc score
#     roc_score = roc_auc_score(target_list.tolist(), score_list.tolist())
#
#     # plot the roc curve
#     if plot_roc_curve:
#         fpr, tpr, _ = roc_curve(target_list.tolist(), score_list.tolist())
#         plt.plot(fpr, tpr, label="Area under ROC = {:.4f}".format(roc_score))
#         plt.legend(loc='best')
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.show()
#
#     # put together values
#     metrics_dict = {
#                     "Accuracy": accuracy,
#                     "Sensitivity": sensitivity,
#                     "Specificity": specificity,
#                     "Roc_score": roc_score,
#                     "Confusion Matrix": conf_matrix,
#                     "Validation Loss": val_loss / len(test_loader),
#                     "score_list": score_list.tolist(),
#                     "pred_list": pred_list.tolist(),
#                     "target_list": target_list.tolist(),
#                     "paths": path_list}
#
#     return metrics_dict