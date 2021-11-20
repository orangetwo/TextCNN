import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


def trainer(model, args, trainIter, testIter):
    device = args.device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    ce_loss = nn.CrossEntropyLoss()

    total_iter = 0

    for e in range(args.epochs):
        for batch in trainIter:
            total_iter += 1
            model.train()
            optimizer.zero_grad()
            feature, labels = tuple(t.to(device) for t in batch)

            logits = model(feature)
            loss = ce_loss(logits, labels)

            loss.backward()
            optimizer.step()

            if total_iter % 100 == 0:
                # TODO: attention to save the model.
                model.eval()

                preds = []
                turth = []
                for batch4test in testIter:
                    feature, labels = tuple(t.to(device) for t in batch4test)

                    with torch.no_grad():
                        logits = model(feature)
                        preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
                        turth.extend(labels.detach().cpu().tolist())

                acc = accuracy_score(turth, preds)
                print(f"test accuracy : {acc}")
