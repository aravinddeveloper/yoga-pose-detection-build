import json
import logging
import os
import pickle
import tarfile
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

MODEL_PATH = "/opt/ml/model/model.tar.gz"
EVALUATION_PATH = "/opt/ml/model/eval_model/"

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_model(model_dir):
    model = VGG(make_layers(cfg=cfg))
    # model.load_state_dict(model_zoo.load_url("https://download.pytorch.org/models/vgg16-397923af.pth"))
    model.classifier = nn.Sequential(nn.Linear(25088, 512),
                                     nn.ReLU(),
                                     nn.Dropout(),
                                     nn.Linear(512, 512),
                                     nn.ReLU(),
                                     nn.Dropout(),
                                     nn.Linear(512, 5))
    model.load_state_dict(torch.load(f'{model_dir}/model.pth', map_location=torch.device('cpu')))
    # model = torch.load('model.pth', map_location=torch.device('cpu'))
    model = model.eval()
   
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-data-path", type=str, default="/opt/ml/processing/data/eval")

if __name__ == "__main__":
    args = parse_args()
    with tarfile.open(MODEL_PATH) as tar:
        tar.extractall(EVALUATION_PATH)
        
    model = get_model(EVALUATION_PATH)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[ 0.229, 0.224, 0.225])])
    out_class = {0: 'downdog', 1: 'goddess', 2: 'plank', 3: 'tree', 4: 'warrior2'}
    true_label = []
    pred_label = []
    for image_file in glob.glob(args.evaluation_data_path+"/*.*")
        true_label.append(image_file.split("/")[-1].split("_")[0])
        normalized = preprocess(Image.open(image_file))
        batchified = normalized.unsqueeze(0)

        # predict
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batchified = batchified.to(device)
        output = model.forward(batchified)
        prob = F.softmax(output,dim=1)
        val,idx = prob.topk(k=1,dim=1)
        pred_label.append(idx.cpu().detach().numpy()[0][0])
        print(prob)
    
    acc = accuracy_score(true_label, pred_label)
    # auc = roc_auc_score(true_label, pred_label)
    # matrix = confusion_matrix(true_label, pred_label)
    
    report_dict = {
        "multiclass_classification_metrics":{
            "accuracy" : {
                "value" : acc,
                "standard_deviation" : "NA"
            }
        }
    
    print("Classification report:\n{}".format(report_dict))

    evaluation_output_path = os.path.join(EVALUATION_PATH, "evaluation.json")
    print("Saving classification report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))