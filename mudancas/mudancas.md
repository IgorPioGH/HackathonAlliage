Principais mudanças
EPOCHS = 120
EARLY_STOPPING_PATIENCE = 20
entrada externa continua [B, 1, H, W]
dentro da U-Net vira [B, 3, H, W]
canal 0 = radiografia
canal 1 = coordenada X entre -1 e +1
canal 2 = coordenada Y entre -1 e +1
CrossEntropy + Focal Tversky
Tversky penaliza mais falso negativo: FP=0.30, FN=0.70
melhor checkpoint continua sendo escolhido pelo Dice de validação
teste continua sendo usado apenas no final

O Dataset ainda entrega:

[B, 1, 256, 512]

e o próprio modelo cria:

radiografia
      +
      X
      +
      Y
      ↓
[B, 3, 256, 512]

Isso permite que a CNN aprenda, por exemplo, que um padrão visual localizado no quadrante superior direito possui significado diferente do mesmo padrão em outra posição. Essa é justamente a propriedade que queremos explorar porque as classes do seu COCO representam dentes específicos, não simplesmente “dente versus fundo”.

Quando esse treinamento terminar, o número que eu mais quero ver é o bloco:

RESULTADO FINAL — U-NET + COORDENADAS + FOCAL TVERSKY

Melhor época:
Épocas executadas:
Melhor Val Dice:
Test Loss:
Test Dice macro:
Test IoU macro:
Test Pixel Accuracy:

Se, por exemplo, aparecer Melhor época: 94, então aumentar de 60 para 120 provavelmente foi importante. Se aparecer Melhor época: 31 e Early Stopping na 51`, sabemos que simplesmente aumentar ainda mais as épocas provavelmente não é o gargalo; aí a próxima melhoria deve atacar resolução ou representação dos dados, e não tempo de treinamento.

