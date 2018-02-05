import sys
sys.path.append("C:\\Users\\indra\\Documents\\Signaturte\\signature_neural\\SigTools")
#print(sys.path)

from SigTools import save_features, train_network, test_signature, predict_with_mlp, predict_with_svm

save_features("./dataset/TrainingSet/")
train_network("./dataset/TrainingSet/")
test_signature("./dataset/TestSet/Questioned/")
predict_with_mlp()
#predict_with_svm()
