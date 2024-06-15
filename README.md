# Self Supervised Learning (SSL) for Human Activity 
Este trabalho emprega técnicas avançadas de aprendizado auto-supervisionado (SSL) para treinar um modelo destinado à classificação das atividades humanas com base nos dados capturados por sensores de smartphones, como acelerômetro e giroscópio. O SSL permite que o modelo aprenda representações significativas diretamente dos dados não rotulados coletados dos sensores. Para entender um pouco mais sobre SSL, verifique o final deste README,

## Estrutura
Este projeto possui a seguinte estrutura:

- `data/`: contém os dados usados no projeto. Seus dados devem ser armazenados aqui. Este diretório pode conter grandes conjuntos de dados e não deve ser adicionado ao repositório Git.
- `data_modules/`: contém implementações de classes `LightningDataModule`. Essas classes são responsáveis por carregar e pré-processar os dados, além de dividir os dados em conjuntos de treinamento, validação e teste.
- `models/`: contém implementações de classes `LightningModule`. Essas classes são responsáveis por definir a arquitetura do modelo e implementar os métodos `forward`, `training_step` e `configure_optimizers`.
- `transforms/`: contém implementações de transformações Numpy/PyTorch. Essas transformações são usadas para pré-processar os dados antes de alimentá-los ao modelo.
- `best_results/`: contém os scripts para pré-treinar o modelo base, treinar o modelo secundário e avaliar o modelo secundário com os melhores hiperparâmetros (por exemplo, taxa de aprendizado) para cada técnica. Os scripts neste diretório devem seguir a API definida pelo professor.
- `report_results/`: contém os scripts para gerar os resultados apresentados no relatório técnico.

## Notebooks
Os arquivos `ipynb` são demonstrativos de todo o processo de treinamento e transformações. Dentre eles, nos temos:
- `har_cnn1d_training.ipynb`: apresenta todo o processo de treinamento da tarefa de pretexto e também para a tarefa de downstream.
- `har_transformations.ipynb`: apresenta todas as transformações realizadas que foram utilizadas na tarefa de pretexto.
- `required_packages.ipynb`: realiza a instalação de todos os pacotes necessário para a execução do projeto.
- `report_results/har_cnn.ipynb`: gera uma serie de métricas e gráficos avaliativos.

## Instalação
Utilizamos Contêineres do VSCode para executar o projeto. Para rodar o projeto, você precisa ter Docker e VSCode instalados em sua máquina.
Se você não sabe como usar Contêineres no VSCode, pode seguir as instruções no [seguinte link](https://github.com/otavioon/container-workspace).

Dentro do contêiner, você pode instalar as dependências executando o seguinte comando:

```bash
pip install -r requirements.txt
```

## Autor
- [Gabriel Rosa](https://github.com/Sr-Souza-dev)

## Self Supervised Learning (SSL)
O desafio de obter grandes volumes de dados rotulados para treinar modelos de aprendizado de máquina é crucial para o desempenho final desses modelos. No entanto, rotular dados pode ser trabalhoso e caro, levando ao desenvolvimento de técnicas como o aprendizado auto-supervisionado (SSL).

No SSL, os modelos são treinados inicialmente em dados não rotulados através de tarefas de pretexto. Essas tarefas são projetadas para extrair automaticamente características úteis dos dados, sem depender de rótulos externos. Por exemplo, um modelo pode ser treinado para prever a próxima palavra em uma sequência de texto, reconstruir uma parte oculta de uma imagem, ou prever a rotação aplicada a uma imagem.

Após essa fase de pré-treinamento, parte do modelo (o "backbone") pode ser transferido e ajustado para uma tarefa específica de interesse (a tarefa de downstream), como classificação de imagens ou análise de sentimentos. Este processo beneficia-se da representação aprendida durante o pré-treinamento, que captura características úteis dos dados de entrada de forma geral.

Essa abordagem visa iniciar o treinamento com uma inicialização melhorada do modelo, potencialmente reduzindo a necessidade de grandes quantidades de dados rotulados desde o início. Isso não apenas economiza recursos de anotação manual, mas também pode resultar em modelos mais robustos e eficazes em uma variedade de tarefas de aprendizado de máquina.

## Licença
Este projeto está licenciado sob a Licença GPL-3.0 - consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
