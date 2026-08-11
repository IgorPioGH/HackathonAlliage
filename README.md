# HackathonAlliage
Nome: Igor Ferronato Fonseca Pio
Nome: Taiane Lopes

## Descrição
Case apresentado no Hackathon patrocinado pela Alliage realizado no dia 11/08/2026.
O desafio escolhido foi o 1:
1. O problema
Hoje, quando um dentista precisa delimitar uma estrutura numa radiografia panorâmica — o contorno de um dente, um canal, uma lesão — ele faz isso à mão, no software, clique por clique. Leva minutos por imagem. Pior: dois profissionais delimitam a mesma estrutura de formas diferentes, e não há como comparar dois exames do mesmo paciente com critério estável.
O que queremos é um modelo que receba a radiografia e devolva a delimitação automaticamente, com qualidade medida. Não precisa ser perfeito — precisa ser consistente e precisa ter um número que diga o quanto é bom.
Isso não é exercício acadêmico. É a base de qualquer recurso de apoio a diagnóstico que a Alliage queira colocar num equipamento de imagem: sem segmentação confiável, não há medição automática, não há comparação entre exames e não há alerta de achado.

## Resultado
O resultado desenvolvido foi o seguinte código, disponível no repositório ou no Google Colab através do link: https://colab.research.google.com/drive/1d5kGjejJAP0l1JZxIjhEWzaydeTsWujT?usp=sharing

**Para executar o código é importante que os dados sejam carregados na seguinte estrutura:**

dataset\
|
| _ _  train\
        |
        |
        | __ images
        |
        |
        |__ annotations

_____ test\
      |
      |__ images
