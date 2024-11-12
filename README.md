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
2. Colocar los archivos en `data/miniImageNet/images`.
3. Ejecutar el script de preparación: `scripts/prepare_mini_imagenet.py`

#### YCBO

1. Descargar los archivos desde [este enlace](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/).

# Resultados

En `experiments/experiments.txt` se incluyen los comandos ejecutar para obtener los resultados.

### Matching Networks

|                     | Omniglot|
|---------------------|---------|
| **k-way**           | **5**   |
| **n-shot**          | **1**   |
| Publicado (cosine)  | -    |
| Repositorio (l2)    | -    |

|                        | miniImageNet|
|------------------------|-------------|
| **k-way**              | **5**       |
| **n-shot**             | **1**       |
| Published (cosine, FCE)| -        |
| Repositorio (l2)       | -        |

### Redes prototípicas


|                  | Omniglot |
|------------------|----------|
| **k-way**        | **5**    |
| **n-shot**       | **1**    |
| Publicado        | -     |
| Repositorio      | -     |

|                  | miniImageNet|
|------------------|-------------|
| **k-way**        | **5**       |
| **n-shot**       | **1**       |
| Publicado        | -        |
| Repositorio      | -        |

|                  | YCBO     |
|------------------|----------|
| **k-way**        | **5**    |
| **n-shot**       | **1**    |
| Repositorio      | -     |

### MAML

|                            |   Omniglot  |
|----------------------------|-------------|
| **k-way**                  | **5**       |
| **n-shot**                 | **1**       |
| Publicado                  | -        |
| Repositorio (Primer orden) | -        |

|                            | miniImageNet|
|----------------------------|-------------|
| **k-way**                  | **5**       |
| **n-shot**                 | **1**       |
| Publicado                  | -       |
| Repositorio (Primer orden) | -       |

### SNAIL


|                  | Omniglot |
|------------------|----------|
| **k-way**        | **5**    |
| **n-shot**       | **1**    |
| Publicado        | -     |
| Repositorio      | -     |

|                  | miniImageNet|
|------------------|-------------|
| **k-way**        | **5**       |
| **n-shot**       | **1**       |
| Publicado        | -        |
| Repositorio      | -        |
