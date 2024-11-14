import torch
from torch.utils.data import DataLoader
from torch.xpu import device

from Transformer import Transformer
from datasets import load_dataset
from transformers import AutoTokenizer
from nltk.translate.bleu_score import corpus_bleu

de_tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
en_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def collate_fn(batch):
    src = []
    tgt = []
    for data in batch:
        en = torch.tensor(en_tokenizer.encode(data['translation']['en'], truncation=True, max_length=512, padding='max_length')).unsqueeze(0)
        de = torch.tensor(de_tokenizer.encode(data['translation']['de'], truncation=True, max_length=512, padding='max_length')).unsqueeze(0)
        src.append(en)
        tgt.append(de)
    src = torch.cat(src, dim=0)
    tgt = torch.cat(tgt, dim=0)
    return {'src': src, 'tgt': tgt}


def train_function(model, train_loader, val_loader, criterion, opt, device, epochs=10):
    min_val_loss = float('inf')
    min_val_epoch = 0
    for epoch in range(epochs):
        model.train()
        train_loss = []
        for idx, batch in enumerate(train_loader):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            output = model(src, tgt[:, :-1])
            output = output.reshape(-1, output.shape[2])
            tgt = tgt[:, 1:].reshape(-1)
            opt.zero_grad()
            loss = criterion(output, tgt)
            loss.backward()
            opt.step()
            train_loss.append(loss.item())
            if idx % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {idx}, Loss: {loss.item()}")
        print(f"Epoch: {epoch}, Train Loss: {sum(train_loss) / len(train_loss)}")

        model.eval()
        with torch.no_grad():
            val_loss = []
            for idx, batch in enumerate(val_loader):
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                output = model(src, tgt[:, :-1])
                output = output.reshape(-1, output.shape[2])
                tgt = tgt[:, 1:].reshape(-1)
                loss = criterion(output, tgt)
                val_loss.append(loss.item())
            print(f"Epoch: {epoch}, Val Loss: {sum(val_loss) / len(val_loss)}")
            if sum(val_loss) / len(val_loss) < min_val_loss:
                min_val_loss = sum(val_loss) / len(val_loss)
                torch.save(model.state_dict(), f"model_pth/transformer.pth")
                min_val_epoch = epoch
                with open("model_pth/transformer.txt", "w") as f:
                    f.write(f"Min Val Loss: {min_val_loss}, Min Val Epoch: {min_val_epoch}")


def evaluate_function(model, test_loader, device):
    model.load_state_dict(torch.load("model_pth/transformer.pth"))
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)

            start_tgt = torch.tensor([de_tokenizer.cls_token_id] * src.shape[0]).unsqueeze(1).to(device)
            while start_tgt.shape[1] < 512 or (start_tgt[:, -1] != de_tokenizer.sep_token_id).any():
                output = model(src, start_tgt)
                output = output[:, -1, :].argmax(dim=1).unsqueeze(1)
                start_tgt = torch.cat([start_tgt, output], dim=1)

            # TODO: Implement BLEU Score







if __name__ == '__main__':

    ds = load_dataset("wmt/wmt14", "de-en")
    train_set = ds['train']
    val_set = ds['validation']
    test_set = ds['test']
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Transformer(src_vocab_size=len(en_tokenizer.get_vocab()), tgt_vocab_size=len(de_tokenizer.get_vocab()),
                        embed_size=512, num_layers=6, num_heads=8, ff_hidden_size=2048, dropout=0.1, device=device,
                        src_pad_idx=en_tokenizer.pad_token_id, tgt_pad_idx=de_tokenizer.pad_token_id)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=de_tokenizer.pad_token_id).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_function(model, train_loader, val_loader, criterion, opt, device, epochs=10)
    evaluate_function(model, test_loader, device)