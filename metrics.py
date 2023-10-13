import pandas as pd
import zipfile
import os, time, requests

from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, balanced_accuracy_score, matthews_corrcoef, f1_score
import matplotlib.pyplot as plt

import json

bt = ""
def fetch_detail_metric(url, file_name, audioIds, bearerToken):
  
  ids = file_name.split('_id_')[1].split('_')
  companyId = ids[0]
  clientId = ids[1]
  areaId = ids[2]
  projectId = ids[3]
  batchId = ids[4]
  data = {
        "companyId": companyId,
        "clientId": clientId,
        "batchId": batchId,
        "audioIds": audioIds
    }
  headers = {"Authorization": 'Bearer '+bearerToken}
  res = requests.post(url = url, json = data, headers=headers)
  return res.json()

def generate_resumed_historial( manual_result_path, conversation_historial_path, resumed_historial_path):
    manual_result = pd.read_excel(manual_result_path)
    conversation_historial = pd.read_excel(conversation_historial_path)

    resumed_historial = pd.merge(left=manual_result, right=conversation_historial, left_on='id_conversation', right_on='idConversation', how='outer')
    resumed_historial = resumed_historial.drop(columns=manual_result.columns)
    resumed_historial.to_excel(resumed_historial_path, index=False)

    print("/nResumed historial generated!/n")

def pretty_name(text):
    words = text.split(" ")
    total_string = ""
    for counter, word in enumerate(words):
        if counter > 0 and counter % 4 == 0:
            total_string += "\n{}".format(word)
        else:
            total_string += " {}".format(word)
    return total_string.strip()

def run(base_path, manual_result_path, automatic_result_path):
    if not os.path.exists(f'{base_path}/confusion_matrices'):
        os.makedirs(f'{base_path}/confusion_matrices')

    manual_result = pd.read_excel(f"{base_path}/{manual_result_path}")
    automatic_result = pd.read_excel(f"{base_path}/{automatic_result_path}")

    res = open(f'{base_path}/confusion_matrices/acc_report.txt', 'w')
    res.write(str(len(set(manual_result['id_conversation']).intersection(set(automatic_result['id_conversation'])))) + " conversaciones en comun\n\n")

    # Common Ids
    common_id=set(manual_result['id_conversation']).intersection(set(automatic_result['id_conversation']))

    manual_result=manual_result[manual_result['id_conversation'].isin(common_id)]
    automatic_result=automatic_result[automatic_result['id_conversation'].isin(common_id)]

    manual_result.sort_values(by='id_conversation',inplace=True)
    automatic_result.sort_values(by='id_conversation',inplace=True)

    confusion_matrix_dict = {}
    fig = plt.figure(figsize=[20,15])
    plt.subplots_adjust(hspace=1)
    cols1 = manual_result.columns.tolist()
    cols2 = automatic_result.columns.tolist()

    # Find the common columns
    interest_columns = list(set(cols1) & set(cols2))
    interest_columns.remove('id_conversation')

    # Create Excel
    extraInfo = fetch_detail_metric('https://development-sa-api.ressolve.ai/operation/opProcess/nlpOpsReporting', automatic_result_path, list(common_id), bearerToken=bt)
    excelDF = pd.DataFrame(columns=['ID','Manual','Automatic', 'Metric', 'originalTranscription', "campaignInfo", "sentimentInfo", "originalNlpInfo"])
    
    for col in interest_columns:
        for id in common_id:
            new_row = {'ID':id, 
                       'Manual':manual_result.loc[manual_result['id_conversation'] == id, col].values[0], 
                       'Automatic':automatic_result.loc[automatic_result['id_conversation'] == id, col].values[0], 
                       'Metric':col,
                       'originalTranscription': next((obj['originalTranscription'] for obj in extraInfo if obj['id'] == id), None),
                       'campaignInfo': next((obj['campaignInfo'] for obj in extraInfo if obj['id'] == id), None),
                       'sentimentInfo': next((obj['sentimentInfo'] for obj in extraInfo if obj['id'] == id), None),
                       'originalNlpInfo': next((obj['originalNlpInfo'] for obj in extraInfo if obj['id'] == id), None)
                       }
            excelDF.loc[len(excelDF)] = new_row
    excelDF.to_excel(f'{base_path}/confusion_matrices/metrics_report.xlsx', index=False)   
    print('Excel created')
    res.write(str(len(interest_columns)) + " confusion matrices\n\n")
    total_accuracy = 0

    # Create a list to store confusion matrices and their titlesbase_path = os.path.dirname(os.path.abspath(__file__))
    confusion_matrices = []

    # Iterate over groups of 10 columns
    for i in range(0, len(interest_columns), 10):
        group_cols = interest_columns[i:i+10]
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))
        for j, col in enumerate(group_cols):
            y_true = manual_result[col]
            y_pred = automatic_result[col]
            accuracy = accuracy_score(y_true, y_pred)
            total_accuracy += accuracy
            confusion_matrix_dict[col] = [sum(y_true), sum(y_pred)]
            title = pretty_name(f'{col} -> {accuracy:1.3f}')
            ax = axes[j // 5, j % 5]
            ConfusionMatrixDisplay.from_predictions(y_true, y_pred, colorbar=False, ax=ax, xticks_rotation='horizontal')
            ax.set_title(title, fontweight="bold")
            res.write(str(i+j+1) + ". " + col + '\n')
            res.write("Image group index -> " + str(i//10) + '\n')
            res.write('Classification report:\n')
            res.write(classification_report(y_true, y_pred, zero_division=0))
            res.write('Accuracy:\n')
            res.write(str(accuracy))
            res.write('\n\n\n')
        plt.tight_layout()
        plt.savefig(f'{base_path}/confusion_matrices/confusion_matrix_group_{i//10}.png')
        plt.close()
        confusion_matrices.clear()
    res.write("Total accuracy: " + str(total_accuracy/len(interest_columns)))
    res.close()
    
    # Save acc_report.txt and confusion images into a zip file
    with zipfile.ZipFile(f'{base_path}/results.zip', 'w') as zipObj:
        # Write acc_report.txt into the zip
        zipObj.write(f'{base_path}/confusion_matrices/acc_report.txt', 'acc_report.txt')
        # Write metrics_report.txt into the zip
        zipObj.write(f'{base_path}/confusion_matrices/metrics_report.xlsx', 'metrics_report.xlsx')
        # Write confusion images into the zip
        for i in range(0, len(interest_columns), 10):
            zipObj.write(f'{base_path}/confusion_matrices/confusion_matrix_group_{i//10}.png', f'confusion_matrix_group_{i//10}.png')

base_path = os.path.dirname(os.path.abspath(__file__))
manual_result_path      = "manual.xlsx"
automatic_result_path   = ".xlsx"

run(base_path, manual_result_path, automatic_result_path)