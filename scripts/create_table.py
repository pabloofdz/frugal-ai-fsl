import io
import pandas as pd
import matplotlib.pyplot as plt

# Datos CSV completos
csv_data = '''epoch,categorical_accuracy,loss,lr,val_1-shot_5-way_acc,val_loss
'''

# Leer los datos usando pandas
df = pd.read_csv(io.StringIO(csv_data))

custom_pink = (177/255, 0/255, 114/255)
darker_pink = (255/255, 101/255, 56/255)
light_blue = (173/255, 216/255, 230/255)
darker_blue = (56/255, 255/255, 96/255)

# Gráfico de Accuracy
plt.figure(figsize=(12, 8))
plt.plot(df['epoch'], df['categorical_accuracy'], label='Exactitud en el entrenamiento', color=light_blue)
plt.plot(df['epoch'], df['val_1-shot_5-way_acc'], label='Exactitud en la evaluación', color=custom_pink)
plt.xlabel('Época', fontsize=22)
plt.ylabel('Exactitud', fontsize=22)
plt.legend(fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)
plt.show()

# Gráfico de Loss
plt.figure(figsize=(12, 8))
plt.plot(df['epoch'], df['loss'], label='Pérdida en el entrenamiento', color=darker_blue)
plt.plot(df['epoch'], df['val_loss'], label='Pérdida en la evaluación', color=darker_pink)
plt.xlabel('Época', fontsize=22)
plt.ylabel('Pérdida', fontsize=22)
plt.legend(fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.grid(True)
plt.show()
