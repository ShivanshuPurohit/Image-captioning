import torch
import torch.nn as nn
import torchvision.models as models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        #save parameters
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        #define the model
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bias=True,
                           dropout=0,
                           batch_first=True)
        self.linear = nn.Linear(hidden_size,vocab_size)
        self.softmax = nn.Softmax(dim=2)
        
    def init_hidden(self, batch_size):
        return (torch.zeros((1, batch_size, self.hidden_size), device=device),
                torch.zeros((1, batch_size, self.hidden_size), device=device))
    
    def forward(self, features, captions):
        captions_no_end = captions[:,:-1]
        captions = self.word_embedding(captions_no_end)
        batch_size = features.shape[0]
        self.hidden = self.init_hidden(batch_size)
        
        inputs = torch.cat((features.unsqueeze(1), captions), dim=1)
        lstm_output, self.hidden = self.lstm(inputs, self.hidden)
        outputs = self.linear(lstm_output)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        res = []
        batch_size = inputs.shape[0]
        hidden = self.init_hidden(batch_size)
        
        while True:
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.linear(lstm_out)
            outputs = outputs.squeeze(1)
            _, predict_index = torch.max(outputs, dim=1)
            res.append(predict_index.cpu().numpy()[0].item())
            
            if predict_index == 1:
                break
                
            inputs = self.word_embedding(predict_index)
            inputs = inputs.unsqueeze(1)
            
        return res