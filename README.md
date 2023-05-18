[Voltar para o repositório principal :house:](https://github.com/rmnicola/m6-ec-encontros.git)

# Detecção de objetos com Python<!-- omit in toc -->

## Objetivos do encontro

## Conteúdo <!-- omit in toc -->
- [Objetivos do encontro](#objetivos-do-encontro)
- [Filtros convolucionais para detecção de objetos](#filtros-convolucionais-para-detecção-de-objetos)
  - [Correlação cruzada vs Convolução](#correlação-cruzada-vs-convolução)
  - [Haar Cascade](#haar-cascade)
- [YoLo](#yolo)
  - [Instalação](#instalação)
  - [Uso](#uso)
  - [Retreinando os pesos da YoLo](#retreinando-os-pesos-da-yolo)
  - [Detecção de objetos usando YoLo](#detecção-de-objetos-usando-yolo)
  - [Segmentação de imagens usando YoLo](#segmentação-de-imagens-usando-yolo)

## Filtros convolucionais para detecção de objetos

Os filtros de convolução são amplamente utilizados na visão computacional para a detecção de objetos em imagens. No processo de desenvolvimento de máscaras, os filtros são projetados para extrair características específicas dos objetos de interesse. Essas máscaras são aplicadas por meio de uma operação de convolução, onde cada elemento da imagem é multiplicado pelos coeficientes da máscara e somados para obter um valor de saída. Esses valores de saída são então comparados com um limiar para determinar a presença ou ausência de um objeto na região analisada.

### Correlação cruzada vs Convolução

![Imagem CorrConv](https://mblogthumb-phinf.pstatic.net/20140419_70/jinohpark79_13979171194788suzT_GIF/crosscorrelation6.GIF?type=w2)

Imagem retirada de [blog naver](https://m.blog.naver.com/jinohpark79/110189322799)

A correlação cruzada é uma técnica fundamental utilizada na detecção de objetos por meio da aplicação de máscaras, também conhecidas como filtros ou núcleos. Nesse procedimento, a correlação cruzada é aplicada para estabelecer uma correspondência entre um filtro, que representa uma versão simplificada do objeto de interesse, e uma região selecionada de uma imagem de entrada. O processo consiste em calcular a similaridade entre os pixels do filtro e os pixels correspondentes da região da imagem, o que resulta em uma pontuação de similaridade para cada posição possível do objeto na imagem. Essa pontuação, em seguida, auxilia na determinação da localização mais provável do objeto na imagem, facilitando a sua detecção e localização precisa.

A correlação cruzada e a convolução são duas operações intimamente relacionadas, amplamente empregadas no processamento de sinais e imagens. Ambas operações consistem na aplicação de um filtro à imagem para obter informações pertinentes. A distinção crucial entre as duas reside na maneira como o filtro é aplicado à imagem. Na convolução, o filtro é virado, ou seja, é espelhado antes de sua aplicação à imagem. Em contrapartida, na correlação cruzada, o filtro é usado como está, sem a necessidade de ser espelhado. Isso resulta em diferentes aplicações práticas: a convolução é comumente utilizada para operações de filtragem, como a detecção de bordas e suavização de imagens, enquanto a correlação cruzada é frequentemente empregada em tarefas de correspondência de padrões, incluindo a detecção de objetos e o reconhecimento de padrões. Mesmo com essa diferença, ambas operações compartilham semelhanças matemáticas e são frequentemente implementadas de maneira similar em diversas bibliotecas e frameworks de processamento de imagens.

### Haar Cascade

![Haar Cascade](https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/haar.png)

Ilustração do processo de detecção de faces utilizando Haar cascade. Imagem retirada da [documentação do opencv](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html)

A detecção de faces usando Haar Cascade é um método eficaz de detecção de objetos que foi proposto por Paul Viola e Michael Jones em 2001. O método é baseado no conceito de 'features' de Haar, que são essencialmente somas e diferenças ponderadas de intensidades de pixel em regiões retangulares adjacentes de uma imagem. O classificador Haar Cascade é treinado por um processo iterativo em que características de Haar são usadas para detectar faces em imagens de treinamento. Esse método seleciona automaticamente as melhores características de uma grande quantidade de potenciais candidatas, utilizando um algoritmo de aprendizado de máquina chamado Adaboost.

O processo de detecção de faces com Haar Cascades é uma aplicação de conceitos de visão computacional clássica. Assim como na detecção de objetos em visão computacional, o método Haar Cascade envolve a extração de características importantes de imagens para discriminar entre diferentes classes de objetos - neste caso, faces e não-faces. Para fazer isso, a imagem é varrida em diferentes tamanhos e orientações para detectar faces de vários tamanhos e em diferentes orientações. A detecção é feita passando a imagem por uma série de classificadores em cascata, começando pelos mais simples e chegando aos mais complexos. Se a imagem passar por todos os classificadores da cascata, ela é considerada uma face.

![Mascaras Haar](https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/haar_features.jpg)

Exemplo de máscaras utilizadas no método Haar Cascade. Imagem retirada da [documentação do opencv](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html)

No entanto, apesar de sua eficácia, a detecção de faces usando Haar Cascades é considerada obsoleta em relação aos métodos modernos de visão computacional. Isso se deve ao surgimento das Redes Neurais Convolucionais (CNNs), que têm uma capacidade muito maior de aprender características discriminativas complexas de imagens. As CNNs podem ser treinadas com um conjunto de dados muito maior e podem aprender a detectar uma variedade muito mais ampla de faces em diferentes condições de iluminação, orientações e expressões faciais. Além disso, as CNNs também são capazes de detectar outros objetos além de faces, tornando-as muito mais versáteis do que os Haar Cascades. Por essas razões, as Redes Neurais Convolucionais se tornaram o padrão ouro para detecção de objetos em visão computacional.

## YoLo

YOLO, que significa "You Only Look Once" (Você Olha Apenas Uma Vez), é uma popular arquitetura de modelo pré-treinada para detecção de objetos. Ao contrário de outras abordagens de detecção de objetos que processam as imagens várias vezes em diferentes escalas e regiões para identificar objetos, o YOLO faz tudo isso de uma vez, tornando-o extremamente rápido e eficiente. O YOLO reconhece objetos em uma imagem e classifica-os, além de fornecer a localização desses objetos através de "bounding boxes". O modelo foi projetado para ser extremamente rápido e preciso, fazendo dele uma escolha ideal para aplicações em tempo real.

Para criar o YOLO, os pesquisadores primeiro treinam o modelo em um grande conjunto de dados de imagens com objetos já identificados e localizados. A imagem é dividida em uma grade e cada célula na grade é responsável por prever um número fixo de bounding boxes. O modelo aprende a prever tanto a classificação do objeto (ou seja, o tipo de objeto presente) quanto a localização do objeto (como coordenadas e tamanho do bounding box) ao mesmo tempo. Essa abordagem unificada para a detecção de objetos é o que torna o YOLO excepcionalmente rápido em comparação com outras arquiteturas de detecção de objetos.

| Versão | Ano de lançamento | Descrição |
| ------ | ----------------- | --------- |
| YOLO v1 | 2016 | O YOLO original, conhecido como "You Only Look Once", trouxe um novo paradigma para a detecção de objetos. Ele dividiu a imagem de entrada em uma grade SxS e cada célula da grade previa B caixas delimitadoras e as probabilidades de classe para essas caixas delimitadoras.|
| YOLO v2 (YOLO 9000) | 2016 | YOLOv2, também conhecido como YOLO 9000, aprimorou a precisão e a velocidade da detecção do YOLO original. Introduziu o conceito de "âncoras" para lidar melhor com objetos de diferentes tamanhos e formas. Além disso, foi capaz de detectar mais de 9000 classes de objetos diferentes |
| YOLO v3 | 2018 | YOLOv3 introduziu detecções em três escalas, detectando objetos em três tamanhos diferentes de células da grade. Além disso, ele usou três tamanhos de caixas delimitadoras para cada célula, resultando em um total de nove caixas delimitadoras para cada célula da grade. Esta versão também utilizou três âncoras por detecção, ao contrário do YOLOv2 que utilizava cinco.|
| YOLO v4 | 2020 | O YOLOv4 apresentou diversas melhorias em relação às versões anteriores, incluindo a utilização de novas funções de ativação (Mish) e melhorias na velocidade e precisão. Além disso, introduziu o conceito de CSPDarknet53 como a rede espinha dorsal e usou PANet e SAM block para melhorar a detecção de objetos.|
| YOLO v5 | 2020 | O YOLOv5 é uma versão não oficial do YOLO, que se concentra em facilitar a implementação prática do YOLO. Apresentou melhorias na velocidade de inferência e treinamento, bem como na precisão. |
| YOLO v7 | 2022 | O YOLOv7 é a versão mais recente e foi introduzido para melhorar a velocidade e a precisão do YOLOv4. Ele introduziu várias reformas arquitetônicas, como E-ELAN (Extended Efficient Layer Aggregation Network) e Escala de Modelo Composto para modelos baseados em concatenação. Além disso, ele usou um conjunto de melhorias gratuitas, como a convolução reparametrizada planejada e o ajuste áspero para perda auxiliar e fino para perda principal.|

### Instalação

### Uso

### Retreinando os pesos da YoLo

### Detecção de objetos usando YoLo

### Segmentação de imagens usando YoLo
