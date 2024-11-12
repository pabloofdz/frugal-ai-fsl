# Inteligencia Artificial Frugal: Aprendizaje Automático con datos limitados

En este repositorio se inclue la implementación de 4 técnicas de few-shot learning basadas en metaaprendizaje:
1. Matching networks (Metaaprendizaje basado en métricas)
2. Redes prototípicas (Metaaprendizaje basado en métricas)
3. MAML (Metaaprendizaje basado en optimización)
4. SNAIL (Metaaprendizaje basado en modelos)

# Requisitos

Los requisitos están listados en `requirements.txt`. Deben instalarse con el siguiente comando, preferentemente en un entorno virtual:

pip install -r requirements.txt

### Datos

Debe editarse la variable `DATA_PATH` en `config.py` para indicar la ubicación donde se almacenarán los datasets: Omniglot, MiniImagenet e YCBO.

Después de obtener los datos y ejecutar los scripts de configuración, la estructura de carpetas debería verse así:

```
DATA_PATH/
    Omniglot/
        images_background/
        images_evaluation/
    miniImageNet/
        images_background/
        images_evaluation/
    YCBObjectSet/
        images_background/
        images_evaluation/
```


#### Omniglot

1. Descargar desde [este enlace](https://github.com/brendenlake/omniglot/tree/master/python).
2. Colocar los archivos extraídos en `DATA_PATH/Omniglot_Raw`.
3. Ejecutar el script de preparación: `scripts/prepare_omniglot.py`


#### MiniImageNet

1. Descargar los archivos desde [este enlace](https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view).
2. Colocar los archivos en `DATA_PATH/miniImageNet/images`.
3. Ejecutar el script de preparación: `scripts/prepare_mini_imagenet.py`

#### YCBO

1. Descargar los archivos desde [este enlace](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/).

# Resultados

En `results/` se encuentran los resultados obtenidos. Se incluyen los comandos empleados, la evolución de las métricas durante el entrenamiento y la huella de carbono (no se han incluído los modelos porque se superaba el límite de peso de Git)
Para generar las tablas a partir de los csv de las métricas se puede emplear `/scripts/create_table`

### Matching Networks

|                     | Omniglot|
|---------------------|---------|
| **N-way**           | **5**   |
| **K-shot**          | **1**   |
| Publicado (cosine)  | 98,10   |
| Repositorio (l2)    | 98,60   |

|                        | miniImageNet|
|------------------------|-------------|
| **N-way**              | **5**       |
| **K-shot**             | **1**       |
| Published (cosine, FCE)| 44,20       |
| Repositorio (l2)       | 44,60       |

### Redes prototípicas


|                  | Omniglot |
|------------------|----------|
| **N-way**        | **5**    |
| **K-shot**       | **1**    |
| Publicado        | 98,80    |
| Repositorio      | 99,00    |

|                  | miniImageNet|
|------------------|-------------|
| **N-way**        | **5**       |
| **K-shot**       | **1**       |
| Publicado        | 49,40       |
| Repositorio      | 51,60       |

|                  | YCBO     |
|------------------|----------|
| **N-way**        | **5**    |
| **K-shot**       | **1**    |
| Repositorio      | 70,20    |

### MAML

|                            |   Omniglot  |
|----------------------------|-------------|
| **N-way**                  | **5**       |
| **K-shot**                 | **1**       |
| Publicado                  | 98,70       |
| Repositorio (Primer orden) | 94,50       |

|                            | miniImageNet|
|----------------------------|-------------|
| **N-way**                  | **5**       |
| **K-shot**                 | **1**       |
| Publicado                  | 46,92       |
| Repositorio (Primer orden) | 48,07       |

### SNAIL


|                  | Omniglot |
|------------------|----------|
| **N-way**        | **5**    |
| **K-shot**       | **1**    |
| Publicado        | 99,07    |
| Repositorio      | 98,34    |

|                  | miniImageNet|
|------------------|-------------|
| **N-way**        | **5**       |
| **K-shot**       | **1**       |
| Publicado        | 50,68       |
| Repositorio      | 55,71       |
