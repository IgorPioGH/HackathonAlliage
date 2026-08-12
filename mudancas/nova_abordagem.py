import csv
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from PIL import Image, ImageDraw, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# ============================================================
# CONFIGURAÇÃO
# ============================================================

IMAGES_DIR = Path(
    "data/images"
)

ANNOTATIONS_DIR = Path(
    "data/annotations"
)

OUTPUT_DIR = Path(
    "outputs/unet_coord_focal_tversky"
)


# ============================================================
# DADOS
# ============================================================

IMAGE_SIZE = (
    256,
    512
)

# 0 = background
# 1..32 = dentes
NUM_CLASSES = 33

BASE_CHANNELS = 32

BATCH_SIZE = 4

NUM_WORKERS = 4


# ============================================================
# TREINAMENTO
# ============================================================

# AUMENTADO:
# antes = 60
# agora = 120
#
# Não significa que obrigatoriamente
# treinaremos 120 épocas.
#
# O Early Stopping pode interromper antes.
EPOCHS = 120


LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4


# ============================================================
# DIVISÃO DOS DADOS
# ============================================================

TRAIN_RATIO = 0.70

VAL_RATIO = 0.15

TEST_RATIO = 0.15


SEED = 42


# ============================================================
# LOSS
#
# CrossEntropy + Focal Tversky
# ============================================================

CE_WEIGHT = 0.40

FOCAL_TVERSKY_WEIGHT = 0.60


# Tversky:
#
# TP
# --------------------------------
# TP + alpha * FP + beta * FN
#
# Como beta > alpha,
# estamos penalizando mais falsos negativos.
#
# Isso significa:
# perder um pixel de dente verdadeiro
# é mais penalizado.
TVERSKY_ALPHA_FP = 0.30

TVERSKY_BETA_FN = 0.70


# Focalização.
#
# Valores > 1 fazem classes/erros
# difíceis terem mais influência
# relativa nesta implementação.
FOCAL_GAMMA = 1.33


# ============================================================
# EARLY STOPPING
# ============================================================

# Aumentado porque aumentamos o teto
# de épocas.
EARLY_STOPPING_PATIENCE = 20


# ============================================================
# SCHEDULER
# ============================================================

SCHEDULER_PATIENCE = 5

SCHEDULER_FACTOR = 0.5

MIN_LR = 1e-6


# ============================================================
# GRADIENTE
# ============================================================

MAX_GRAD_NORM = 1.0


# ============================================================
# MIXED PRECISION
# ============================================================

USE_AMP = True


# ============================================================
# AUGMENTATION
# ============================================================

USE_PHOTOMETRIC_AUGMENTATION = True


BRIGHTNESS_RANGE = (
    0.90,
    1.10
)


CONTRAST_RANGE = (
    0.90,
    1.10
)


# ============================================================
# PESOS DA CROSS ENTROPY
# ============================================================

USE_CLASS_WEIGHTS = True


CLASS_WEIGHT_MIN = 0.25

CLASS_WEIGHT_MAX = 5.0


# ============================================================
# VISUALIZAÇÕES
# ============================================================

NUM_PREDICTIONS_TO_PLOT = 3


# ============================================================
# REPRODUTIBILIDADE
# ============================================================

def set_seed(
    seed
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )


# ============================================================
# DATASET
# ============================================================

class DentalPanoramicDataset(
    Dataset
):

    def __init__(
        self,
        images_dir,
        json_files,
        image_size=(256, 512),
        augment=False
    ):

        self.images_dir = Path(
            images_dir
        )

        self.json_files = [

            Path(file)

            for file
            in json_files

        ]

        self.image_size = (
            image_size
        )

        self.augment = (
            augment
        )


        if not self.json_files:

            raise RuntimeError(
                "Nenhum JSON fornecido ao Dataset."
            )


    def __len__(
        self
    ):

        return len(
            self.json_files
        )


    # ========================================================
    # COCO -> MÁSCARA
    # ========================================================

    @staticmethod
    def _create_mask(
        coco,
        image_id,
        width,
        height
    ):

        mask = Image.new(

            mode="L",

            size=(
                width,
                height
            ),

            color=0

        )


        draw = ImageDraw.Draw(
            mask
        )


        for annotation in coco[
            "annotations"
        ]:


            if (
                annotation["image_id"]
                !=
                image_id
            ):

                continue


            class_id = int(
                annotation[
                    "category_id"
                ]
            )


            if not (
                0
                <=
                class_id
                <
                NUM_CLASSES
            ):

                raise ValueError(

                    f"category_id={class_id} "
                    f"fora do intervalo "
                    f"0..{NUM_CLASSES - 1}"

                )


            segmentation = (
                annotation.get(
                    "segmentation",
                    []
                )
            )


            if not isinstance(
                segmentation,
                list
            ):

                continue


            for polygon in segmentation:


                if (
                    not isinstance(
                        polygon,
                        list
                    )
                    or
                    len(polygon) < 6
                ):

                    continue


                points = [

                    (
                        polygon[i],
                        polygon[i + 1]
                    )

                    for i in range(
                        0,
                        len(polygon),
                        2
                    )

                ]


                draw.polygon(

                    points,

                    fill=class_id

                )


        return mask


    # ========================================================
    # AUGMENTATION
    # ========================================================

    @staticmethod
    def _photometric_augmentation(
        image
    ):

        # ------------------------------------------
        # BRILHO
        # ------------------------------------------

        if random.random() < 0.5:

            factor = random.uniform(
                *BRIGHTNESS_RANGE
            )


            image = (
                ImageEnhance
                .Brightness(
                    image
                )
                .enhance(
                    factor
                )
            )


        # ------------------------------------------
        # CONTRASTE
        # ------------------------------------------

        if random.random() < 0.5:

            factor = random.uniform(
                *CONTRAST_RANGE
            )


            image = (
                ImageEnhance
                .Contrast(
                    image
                )
                .enhance(
                    factor
                )
            )


        return image


    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        json_path = (
            self.json_files[
                index
            ]
        )


        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

        with open(

            json_path,

            "r",

            encoding="utf-8"

        ) as file:

            coco = json.load(
                file
            )


        images_info = coco.get(
            "images",
            []
        )


        if len(
            images_info
        ) != 1:

            raise ValueError(

                f"Esperava 1 imagem por JSON, "
                f"mas encontrei "
                f"{len(images_info)} em "
                f"{json_path.name}."

            )


        image_info = (
            images_info[0]
        )


        image_id = (
            image_info[
                "id"
            ]
        )


        image_name = (
            image_info[
                "file_name"
            ]
        )


        image_path = (

            self.images_dir

            /

            image_name

        )


        if not image_path.exists():

            raise FileNotFoundError(

                f"Imagem não encontrada: "
                f"{image_path}"

            )


        # ----------------------------------------------------
        # RADIOGRAFIA
        # ----------------------------------------------------

        image = (
            Image.open(
                image_path
            )
            .convert(
                "L"
            )
        )


        width, height = (
            image.size
        )


        # ----------------------------------------------------
        # VALIDAR DIMENSÕES
        # ----------------------------------------------------

        if (

            width
            !=
            image_info["width"]

            or

            height
            !=
            image_info["height"]

        ):

            raise ValueError(

                f"Dimensão inconsistente em "
                f"{image_name}: "

                f"arquivo="
                f"{width}x{height}, "

                f"JSON="
                f"{image_info['width']}x"
                f"{image_info['height']}"

            )


        # ----------------------------------------------------
        # MÁSCARA
        # ----------------------------------------------------

        mask = self._create_mask(

            coco=coco,

            image_id=image_id,

            width=width,

            height=height

        )


        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        target_height, target_width = (
            self.image_size
        )


        image = image.resize(

            (
                target_width,
                target_height
            ),

            resample=(
                Image
                .Resampling
                .BILINEAR
            )

        )


        mask = mask.resize(

            (
                target_width,
                target_height
            ),

            resample=(
                Image
                .Resampling
                .NEAREST
            )

        )


        # ----------------------------------------------------
        # AUGMENTATION
        # ----------------------------------------------------

        if (

            self.augment

            and

            USE_PHOTOMETRIC_AUGMENTATION

        ):

            image = (
                self
                ._photometric_augmentation(
                    image
                )
            )


        # ----------------------------------------------------
        # PIL -> NUMPY
        # ----------------------------------------------------

        image_np = np.array(

            image,

            dtype=np.float32,

            copy=True

        )


        # 0..255 -> 0..1

        image_np /= 255.0


        mask_np = np.array(

            mask,

            dtype=np.int64,

            copy=True

        )


        # ----------------------------------------------------
        # IMAGEM
        #
        # [H,W]
        #
        # ->
        #
        # [1,H,W]
        #
        # IMPORTANTE:
        #
        # X/Y ainda NÃO entram aqui.
        # Serão criados dentro do modelo.
        # ----------------------------------------------------

        image_np = np.expand_dims(

            image_np,

            axis=0

        )


        image_np = np.ascontiguousarray(
            image_np
        )


        mask_np = np.ascontiguousarray(
            mask_np
        )


        # ----------------------------------------------------
        # TORCH
        # ----------------------------------------------------

        image_tensor = (
            torch
            .from_numpy(
                image_np
            )
            .float()
        )


        mask_tensor = (
            torch
            .from_numpy(
                mask_np
            )
            .long()
        )


        return (
            image_tensor,
            mask_tensor
        )


# ============================================================
# COORDENADAS X / Y
# ============================================================

class AddCoordinateChannels(
    nn.Module
):

    """
    Entrada:

        [B, 1, H, W]

    Saída:

        [B, 3, H, W]


    Canal 0:
        radiografia

    Canal 1:
        coordenada X
        -1 = esquerda
        +1 = direita

    Canal 2:
        coordenada Y
        -1 = topo
        +1 = baixo
    """


    def forward(
        self,
        x
    ):

        (
            batch,
            _,
            height,
            width

        ) = x.shape


        # ====================================================
        # X
        # ====================================================

        x_coords = torch.linspace(

            -1.0,

            1.0,

            steps=width,

            device=x.device,

            dtype=x.dtype

        )


        x_coords = x_coords.view(

            1,

            1,

            1,

            width

        )


        x_coords = x_coords.expand(

            batch,

            1,

            height,

            width

        )


        # ====================================================
        # Y
        # ====================================================

        y_coords = torch.linspace(

            -1.0,

            1.0,

            steps=height,

            device=x.device,

            dtype=x.dtype

        )


        y_coords = y_coords.view(

            1,

            1,

            height,

            1

        )


        y_coords = y_coords.expand(

            batch,

            1,

            height,

            width

        )


        # ====================================================
        # CONCATENAR
        # ====================================================

        output = torch.cat(

            [
                x,
                x_coords,
                y_coords
            ],

            dim=1

        )


        return output


# ============================================================
# DOUBLE CONV
# ============================================================

class DoubleConv(
    nn.Module
):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()


        self.block = nn.Sequential(


            nn.Conv2d(

                in_channels,

                out_channels,

                kernel_size=3,

                padding=1,

                bias=False

            ),


            nn.BatchNorm2d(
                out_channels
            ),


            nn.ReLU(
                inplace=True
            ),


            nn.Conv2d(

                out_channels,

                out_channels,

                kernel_size=3,

                padding=1,

                bias=False

            ),


            nn.BatchNorm2d(
                out_channels
            ),


            nn.ReLU(
                inplace=True
            )

        )


    def forward(
        self,
        x
    ):

        return self.block(
            x
        )


# ============================================================
# U-NET + COORDENADAS
# ============================================================

class UNetWithCoordinates(
    nn.Module
):

    def __init__(
        self,
        num_classes=33,
        base_channels=32
    ):

        super().__init__()


        # ====================================================
        # COORDENADAS
        # ====================================================

        self.add_coordinates = (
            AddCoordinateChannels()
        )


        c1 = (
            base_channels
        )

        c2 = (
            base_channels
            *
            2
        )

        c3 = (
            base_channels
            *
            4
        )

        c4 = (
            base_channels
            *
            8
        )

        c5 = (
            base_channels
            *
            16
        )


        # ====================================================
        # ENCODER
        #
        # ATENÇÃO:
        #
        # agora são 3 canais:
        #
        # imagem
        # X
        # Y
        # ====================================================

        self.enc1 = DoubleConv(
            3,
            c1
        )


        self.pool1 = (
            nn.MaxPool2d(
                2
            )
        )


        self.enc2 = DoubleConv(
            c1,
            c2
        )


        self.pool2 = (
            nn.MaxPool2d(
                2
            )
        )


        self.enc3 = DoubleConv(
            c2,
            c3
        )


        self.pool3 = (
            nn.MaxPool2d(
                2
            )
        )


        self.enc4 = DoubleConv(
            c3,
            c4
        )


        self.pool4 = (
            nn.MaxPool2d(
                2
            )
        )


        # ====================================================
        # BOTTLENECK
        # ====================================================

        self.bottleneck = DoubleConv(
            c4,
            c5
        )


        # ====================================================
        # DECODER 4
        # ====================================================

        self.up4 = (
            nn.ConvTranspose2d(

                c5,

                c4,

                kernel_size=2,

                stride=2

            )
        )


        self.dec4 = DoubleConv(

            c4 + c4,

            c4

        )


        # ====================================================
        # DECODER 3
        # ====================================================

        self.up3 = (
            nn.ConvTranspose2d(

                c4,

                c3,

                kernel_size=2,

                stride=2

            )
        )


        self.dec3 = DoubleConv(

            c3 + c3,

            c3

        )


        # ====================================================
        # DECODER 2
        # ====================================================

        self.up2 = (
            nn.ConvTranspose2d(

                c3,

                c2,

                kernel_size=2,

                stride=2

            )
        )


        self.dec2 = DoubleConv(

            c2 + c2,

            c2

        )


        # ====================================================
        # DECODER 1
        # ====================================================

        self.up1 = (
            nn.ConvTranspose2d(

                c2,

                c1,

                kernel_size=2,

                stride=2

            )
        )


        self.dec1 = DoubleConv(

            c1 + c1,

            c1

        )


        # ====================================================
        # 33 CLASSES
        # ====================================================

        self.final_conv = (
            nn.Conv2d(

                c1,

                num_classes,

                kernel_size=1

            )
        )


    def forward(
        self,
        x
    ):

        # ====================================================
        # [B,1,H,W]
        #
        # ->
        #
        # [B,3,H,W]
        # ====================================================

        x = self.add_coordinates(
            x
        )


        # ====================================================
        # ENCODER
        # ====================================================

        x1 = self.enc1(
            x
        )


        x2 = self.enc2(

            self.pool1(
                x1
            )

        )


        x3 = self.enc3(

            self.pool2(
                x2
            )

        )


        x4 = self.enc4(

            self.pool3(
                x3
            )

        )


        # ====================================================
        # BOTTLENECK
        # ====================================================

        bottleneck = self.bottleneck(

            self.pool4(
                x4
            )

        )


        # ====================================================
        # DECODER 4
        # ====================================================

        d4 = self.up4(
            bottleneck
        )


        d4 = torch.cat(

            [
                d4,
                x4
            ],

            dim=1

        )


        d4 = self.dec4(
            d4
        )


        # ====================================================
        # DECODER 3
        # ====================================================

        d3 = self.up3(
            d4
        )


        d3 = torch.cat(

            [
                d3,
                x3
            ],

            dim=1

        )


        d3 = self.dec3(
            d3
        )


        # ====================================================
        # DECODER 2
        # ====================================================

        d2 = self.up2(
            d3
        )


        d2 = torch.cat(

            [
                d2,
                x2
            ],

            dim=1

        )


        d2 = self.dec2(
            d2
        )


        # ====================================================
        # DECODER 1
        # ====================================================

        d1 = self.up1(
            d2
        )


        d1 = torch.cat(

            [
                d1,
                x1
            ],

            dim=1

        )


        d1 = self.dec1(
            d1
        )


        # ====================================================
        # LOGITS
        # ====================================================

        logits = self.final_conv(
            d1
        )


        return logits


# ============================================================
# FOCAL TVERSKY LOSS
# ============================================================

class MulticlassFocalTverskyLoss(
    nn.Module
):

    def __init__(
        self,
        num_classes,
        alpha_fp=0.30,
        beta_fn=0.70,
        gamma=1.33,
        exclude_background=True,
        smooth=1e-6
    ):

        super().__init__()


        self.num_classes = (
            num_classes
        )


        self.alpha_fp = (
            alpha_fp
        )


        self.beta_fn = (
            beta_fn
        )


        self.gamma = (
            gamma
        )


        self.exclude_background = (
            exclude_background
        )


        self.smooth = (
            smooth
        )


    def forward(
        self,
        logits,
        targets
    ):

        # ====================================================
        # SOFTMAX
        # ====================================================

        probabilities = torch.softmax(

            logits,

            dim=1

        )


        # ====================================================
        # TARGET -> ONE HOT
        #
        # [B,H,W]
        #
        # ->
        #
        # [B,C,H,W]
        # ====================================================

        one_hot = (

            torch
            .nn
            .functional
            .one_hot(

                targets,

                num_classes=(
                    self.num_classes
                )

            )

            .permute(
                0,
                3,
                1,
                2
            )

            .float()

        )


        dims = (
            0,
            2,
            3
        )


        # ====================================================
        # TRUE POSITIVE
        # ====================================================

        true_positive = (

            probabilities

            *

            one_hot

        ).sum(

            dim=dims

        )


        # ====================================================
        # FALSE POSITIVE
        # ====================================================

        false_positive = (

            probabilities

            *

            (
                1.0
                -
                one_hot
            )

        ).sum(

            dim=dims

        )


        # ====================================================
        # FALSE NEGATIVE
        # ====================================================

        false_negative = (

            (
                1.0
                -
                probabilities
            )

            *

            one_hot

        ).sum(

            dim=dims

        )


        # ====================================================
        # TVERSKY INDEX
        # ====================================================

        tversky = (

            true_positive

            +
            self.smooth

        ) / (

            true_positive

            +

            self.alpha_fp
            *
            false_positive

            +

            self.beta_fn
            *
            false_negative

            +

            self.smooth

        )


        # ====================================================
        # QUAIS CLASSES EXISTEM NO BATCH?
        # ====================================================

        present = (

            one_hot.sum(
                dim=dims
            )

            >
            0

        )


        if self.exclude_background:

            present[0] = False


        if not torch.any(
            present
        ):

            return (
                logits.sum()
                *
                0.0
            )


        # ====================================================
        # FOCAL TVERSKY
        # ====================================================

        focal_tversky = (

            1.0

            -

            tversky[
                present
            ]

        ).pow(

            self.gamma

        )


        return (
            focal_tversky.mean()
        )


# ============================================================
# LOSS COMBINADA
# ============================================================

class CombinedLoss(
    nn.Module
):

    def __init__(
        self,
        num_classes,
        class_weights=None
    ):

        super().__init__()


        # ====================================================
        # CROSS ENTROPY
        # ====================================================

        self.ce = nn.CrossEntropyLoss(

            weight=(
                class_weights
            )

        )


        # ====================================================
        # FOCAL TVERSKY
        # ====================================================

        self.focal_tversky = (
            MulticlassFocalTverskyLoss(

                num_classes=(
                    num_classes
                ),

                alpha_fp=(
                    TVERSKY_ALPHA_FP
                ),

                beta_fn=(
                    TVERSKY_BETA_FN
                ),

                gamma=(
                    FOCAL_GAMMA
                ),

                exclude_background=True

            )
        )


    def forward(
        self,
        logits,
        targets
    ):

        ce = self.ce(

            logits,

            targets

        )


        focal_tversky = (
            self.focal_tversky(

                logits,

                targets

            )
        )


        total = (

            CE_WEIGHT
            *
            ce

            +

            FOCAL_TVERSKY_WEIGHT
            *
            focal_tversky

        )


        return (

            total,

            ce.detach(),

            focal_tversky.detach()

        )


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

def update_confusion_matrix(
    confusion,
    predictions,
    targets,
    num_classes
):

    predictions = (

        predictions

        .reshape(-1)

        .to(
            torch.int64
        )

    )


    targets = (

        targets

        .reshape(-1)

        .to(
            torch.int64
        )

    )


    valid = (

        (
            targets >= 0
        )

        &

        (
            targets < num_classes
        )

    )


    encoded = (

        targets[
            valid
        ]

        *
        num_classes

        +

        predictions[
            valid
        ]

    )


    counts = torch.bincount(

        encoded,

        minlength=(
            num_classes
            *
            num_classes
        )

    )


    confusion += (

        counts

        .reshape(

            num_classes,

            num_classes

        )

        .cpu()

    )


    return confusion


# ============================================================
# DICE / IOU / ACCURACY
# ============================================================

def metrics_from_confusion(
    confusion,
    exclude_background=True
):

    confusion = confusion.to(
        torch.float64
    )


    # ========================================================
    # TP
    # ========================================================

    tp = torch.diag(
        confusion
    )


    actual = confusion.sum(
        dim=1
    )


    predicted = confusion.sum(
        dim=0
    )


    fp = (
        predicted
        -
        tp
    )


    fn = (
        actual
        -
        tp
    )


    # ========================================================
    # DICE
    # ========================================================

    dice_denominator = (

        2.0
        *
        tp

        +

        fp

        +

        fn

    )


    dice = torch.where(

        dice_denominator
        >
        0,

        (
            2.0
            *
            tp
        )

        /

        dice_denominator,

        torch.nan

    )


    # ========================================================
    # IOU
    # ========================================================

    iou_denominator = (

        tp

        +

        fp

        +

        fn

    )


    iou = torch.where(

        iou_denominator
        >
        0,

        tp
        /
        iou_denominator,

        torch.nan

    )


    # ========================================================
    # PIXEL ACCURACY
    # ========================================================

    if confusion.sum() > 0:

        pixel_accuracy = (

            tp.sum()

            /

            confusion.sum()

        )

    else:

        pixel_accuracy = (
            torch.tensor(
                float(
                    "nan"
                )
            )
        )


    start = (

        1

        if exclude_background

        else

        0

    )


    macro_dice = torch.nanmean(

        dice[
            start:
        ]

    )


    macro_iou = torch.nanmean(

        iou[
            start:
        ]

    )


    return {

        "pixel_accuracy":
            float(
                pixel_accuracy.item()
            ),

        "dice_macro":
            float(
                macro_dice.item()
            ),

        "iou_macro":
            float(
                macro_iou.item()
            ),

        "dice_per_class":
            dice.cpu().numpy(),

        "iou_per_class":
            iou.cpu().numpy(),

        "support_per_class":
            actual.cpu().numpy()

    }


# ============================================================
# PESOS DE CLASSE
# ============================================================

@torch.no_grad()
def compute_class_weights(
    dataset,
    num_classes
):

    histogram = torch.zeros(

        num_classes,

        dtype=torch.float64

    )


    print(
        "\nCalculando frequência "
        "das classes..."
    )


    for i in tqdm(

        range(
            len(dataset)
        ),

        desc="Class weights"

    ):

        _, mask = dataset[
            i
        ]


        histogram += (

            torch.bincount(

                mask.flatten(),

                minlength=(
                    num_classes
                )

            )

            .double()

        )


    total = histogram.sum()


    frequency = (

        histogram

        /

        total.clamp_min(
            1.0
        )

    )


    # ========================================================
    # PESOS:
    #
    # 1 / sqrt(frequência)
    #
    # menos agressivo que 1/frequência
    # ========================================================

    weights = torch.zeros_like(
        frequency
    )


    present = (
        histogram
        >
        0
    )


    weights[
        present
    ] = (

        1.0

        /

        torch.sqrt(

            frequency[
                present
            ]

            +

            1e-12

        )

    )


    # Média dos pesos = 1

    if torch.any(
        present
    ):

        weights[
            present
        ] /= (

            weights[
                present
            ].mean()

        )


    weights = torch.clamp(

        weights,

        min=(
            CLASS_WEIGHT_MIN
        ),

        max=(
            CLASS_WEIGHT_MAX
        )

    )


    weights[
        ~present
    ] = 0.0


    print(
        "\nPesos das classes:"
    )


    for class_id in range(
        num_classes
    ):

        print(

            f"Classe {class_id:2d} | "

            f"freq="
            f"{frequency[class_id].item():.6f} | "

            f"peso="
            f"{weights[class_id].item():.4f}"

        )


    return weights.float()


# ============================================================
# SPLIT TREINO / VAL / TESTE
# ============================================================

def split_json_files(
    json_files,
    seed=42
):

    json_files = list(
        json_files
    )


    if len(
        json_files
    ) < 3:

        raise RuntimeError(

            "São necessários pelo menos "
            "3 exames."

        )


    rng = random.Random(
        seed
    )


    rng.shuffle(
        json_files
    )


    n = len(
        json_files
    )


    n_train = max(

        1,

        int(
            round(
                n
                *
                TRAIN_RATIO
            )
        )

    )


    n_val = max(

        1,

        int(
            round(
                n
                *
                VAL_RATIO
            )
        )

    )


    if (
        n_train
        +
        n_val
        >=
        n
    ):

        n_train = max(
            1,
            n - 2
        )

        n_val = 1


    n_test = (

        n

        -

        n_train

        -

        n_val

    )


    if n_test < 1:

        n_test = 1

        n_train -= 1


    train_files = (
        json_files[
            :n_train
        ]
    )


    val_files = (
        json_files[

            n_train
            :
            n_train + n_val

        ]
    )


    test_files = (
        json_files[

            n_train + n_val
            :

        ]
    )


    return (

        train_files,

        val_files,

        test_files

    )


# ============================================================
# DATALOADER
# ============================================================

def make_loader(
    dataset,
    shuffle
):

    return DataLoader(

        dataset,

        batch_size=(
            BATCH_SIZE
        ),

        shuffle=(
            shuffle
        ),

        num_workers=(
            NUM_WORKERS
        ),

        pin_memory=(
            torch.cuda.is_available()
        ),

        persistent_workers=(
            NUM_WORKERS > 0
        ),

        drop_last=False

    )


# ============================================================
# TREINAR 1 ÉPOCA
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    amp_enabled
):

    model.train()


    total_loss = 0.0

    total_ce = 0.0

    total_ft = 0.0

    total_samples = 0


    progress = tqdm(

        loader,

        desc="Treino",

        leave=False

    )


    for images, masks in progress:


        # ====================================================
        # CPU -> GPU
        # ====================================================

        images = images.to(

            device,

            non_blocking=True

        )


        masks = masks.to(

            device,

            non_blocking=True

        )


        optimizer.zero_grad(
            set_to_none=True
        )


        # ====================================================
        # FORWARD
        # ====================================================

        with torch.autocast(

            device_type=(
                device.type
            ),

            dtype=(
                torch.float16
            ),

            enabled=(
                amp_enabled
            )

        ):

            logits = model(
                images
            )


            (
                loss,
                ce_loss,
                ft_loss

            ) = criterion(

                logits,

                masks

            )


        # ====================================================
        # BACKPROPAGATION
        # ====================================================

        scaler.scale(
            loss
        ).backward()


        # Antes do clipping,
        # remover scaling do AMP.

        scaler.unscale_(
            optimizer
        )


        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=(
                MAX_GRAD_NORM
            )

        )


        # ====================================================
        # ATUALIZAR PESOS
        # ====================================================

        scaler.step(
            optimizer
        )


        scaler.update()


        batch_size = (
            images.size(
                0
            )
        )


        total_samples += (
            batch_size
        )


        total_loss += (

            loss.item()

            *

            batch_size

        )


        total_ce += (

            ce_loss.item()

            *

            batch_size

        )


        total_ft += (

            ft_loss.item()

            *

            batch_size

        )


        progress.set_postfix(

            loss=(
                f"{loss.item():.4f}"
            ),

            lr=(

                f"{optimizer.param_groups[0]['lr']:.2e}"

            )

        )


    return {

        "loss":
            total_loss
            /
            total_samples,

        "ce_loss":
            total_ce
            /
            total_samples,

        "focal_tversky_loss":
            total_ft
            /
            total_samples

    }


# ============================================================
# AVALIAÇÃO
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    amp_enabled,
    description="Val"
):

    model.eval()


    total_loss = 0.0

    total_ce = 0.0

    total_ft = 0.0

    total_samples = 0


    confusion = torch.zeros(

        (
            NUM_CLASSES,
            NUM_CLASSES
        ),

        dtype=torch.int64

    )


    progress = tqdm(

        loader,

        desc=description,

        leave=False

    )


    for images, masks in progress:


        images = images.to(

            device,

            non_blocking=True

        )


        masks = masks.to(

            device,

            non_blocking=True

        )


        with torch.autocast(

            device_type=(
                device.type
            ),

            dtype=(
                torch.float16
            ),

            enabled=(
                amp_enabled
            )

        ):

            logits = model(
                images
            )


            (
                loss,
                ce_loss,
                ft_loss

            ) = criterion(

                logits,

                masks

            )


        # ====================================================
        # LOGITS -> CLASSE
        # ====================================================

        predictions = torch.argmax(

            logits,

            dim=1

        )


        # ====================================================
        # MATRIZ DE CONFUSÃO
        # ====================================================

        confusion = (
            update_confusion_matrix(

                confusion=confusion,

                predictions=(
                    predictions
                ),

                targets=(
                    masks
                ),

                num_classes=(
                    NUM_CLASSES
                )

            )
        )


        batch_size = (
            images.size(
                0
            )
        )


        total_samples += (
            batch_size
        )


        total_loss += (

            loss.item()

            *

            batch_size

        )


        total_ce += (

            ce_loss.item()

            *

            batch_size

        )


        total_ft += (

            ft_loss.item()

            *

            batch_size

        )


    metrics = (
        metrics_from_confusion(

            confusion,

            exclude_background=True

        )
    )


    metrics.update({

        "loss":
            total_loss
            /
            total_samples,

        "ce_loss":
            total_ce
            /
            total_samples,

        "focal_tversky_loss":
            total_ft
            /
            total_samples,

        "confusion_matrix":
            confusion

    })


    return metrics


# ============================================================
# NOMES DOS DENTES
# ============================================================

def load_category_names(
    json_file
):

    with open(

        json_file,

        "r",

        encoding="utf-8"

    ) as file:

        coco = json.load(
            file
        )


    names = {

        0:
        "background"

    }


    for category in coco.get(
        "categories",
        []
    ):

        names[
            int(
                category["id"]
            )
        ] = str(
            category["name"]
        )


    return names


# ============================================================
# SALVAR SPLIT
# ============================================================

def save_split_files(
    train_files,
    val_files,
    test_files
):

    split_data = {

        "train": [

            str(path)

            for path
            in train_files

        ],

        "val": [

            str(path)

            for path
            in val_files

        ],

        "test": [

            str(path)

            for path
            in test_files

        ]

    }


    with open(

        OUTPUT_DIR
        /
        "split.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            split_data,

            file,

            indent=2,

            ensure_ascii=False

        )


# ============================================================
# SALVAR HISTÓRICO
# ============================================================

def save_history(
    history
):

    if not history:

        return


    path = (

        OUTPUT_DIR

        /

        "history.csv"

    )


    with open(

        path,

        "w",

        newline="",

        encoding="utf-8"

    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=list(
                history[0].keys()
            )

        )


        writer.writeheader()


        writer.writerows(
            history
        )


# ============================================================
# CURVAS
# ============================================================

def save_training_curves(
    history
):

    epochs = [

        row["epoch"]

        for row
        in history

    ]


    fig, axes = plt.subplots(

        1,

        3,

        figsize=(
            18,
            5
        )

    )


    # ========================================================
    # LOSS
    # ========================================================

    axes[0].plot(

        epochs,

        [
            row[
                "train_loss"
            ]

            for row
            in history
        ],

        label="treino"

    )


    axes[0].plot(

        epochs,

        [
            row[
                "val_loss"
            ]

            for row
            in history
        ],

        label="validação"

    )


    axes[0].set_title(
        "Loss"
    )


    axes[0].set_xlabel(
        "Época"
    )


    axes[0].legend()


    # ========================================================
    # DICE
    # ========================================================

    axes[1].plot(

        epochs,

        [
            row[
                "val_dice"
            ]

            for row
            in history
        ]

    )


    axes[1].set_title(
        "Dice — Validação"
    )


    axes[1].set_xlabel(
        "Época"
    )


    # ========================================================
    # IOU
    # ========================================================

    axes[2].plot(

        epochs,

        [
            row[
                "val_iou"
            ]

            for row
            in history
        ]

    )


    axes[2].set_title(
        "IoU — Validação"
    )


    axes[2].set_xlabel(
        "Época"
    )


    fig.tight_layout()


    fig.savefig(

        OUTPUT_DIR
        /
        "training_curves.png",

        dpi=160,

        bbox_inches="tight"

    )


    plt.close(
        fig
    )


# ============================================================
# SALVAR TESTE
# ============================================================

def save_test_results(
    metrics,
    category_names
):

    summary = {

        "test_loss":
            metrics[
                "loss"
            ],

        "test_ce_loss":
            metrics[
                "ce_loss"
            ],

        "test_focal_tversky_loss":
            metrics[
                "focal_tversky_loss"
            ],

        "test_pixel_accuracy":
            metrics[
                "pixel_accuracy"
            ],

        "test_dice_macro":
            metrics[
                "dice_macro"
            ],

        "test_iou_macro":
            metrics[
                "iou_macro"
            ]

    }


    with open(

        OUTPUT_DIR
        /
        "test_metrics.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            summary,

            file,

            indent=2,

            ensure_ascii=False

        )


    # ========================================================
    # MÉTRICAS POR DENTE
    # ========================================================

    with open(

        OUTPUT_DIR
        /
        "per_class_metrics.csv",

        "w",

        newline="",

        encoding="utf-8"

    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow([

            "class_id",

            "class_name",

            "dice",

            "iou",

            "support_pixels"

        ])


        for class_id in range(
            NUM_CLASSES
        ):

            writer.writerow([

                class_id,

                category_names.get(
                    class_id,
                    str(class_id)
                ),

                metrics[
                    "dice_per_class"
                ][class_id],

                metrics[
                    "iou_per_class"
                ][class_id],

                int(
                    metrics[
                        "support_per_class"
                    ][class_id]
                )

            ])


    # ========================================================
    # MATRIZ DE CONFUSÃO
    # ========================================================

    np.savetxt(

        OUTPUT_DIR
        /
        "test_confusion_matrix.csv",

        metrics[
            "confusion_matrix"
        ].numpy(),

        delimiter=",",

        fmt="%d"

    )


# ============================================================
# VISUALIZAR PREDIÇÕES
# ============================================================

@torch.no_grad()
def save_prediction_examples(
    model,
    loader,
    device,
    amp_enabled
):

    model.eval()


    images, masks = next(
        iter(loader)
    )


    images_gpu = images.to(

        device,

        non_blocking=True

    )


    with torch.autocast(

        device_type=(
            device.type
        ),

        dtype=(
            torch.float16
        ),

        enabled=(
            amp_enabled
        )

    ):

        logits = model(
            images_gpu
        )


    predictions = (

        torch.argmax(

            logits,

            dim=1

        )

        .cpu()

    )


    n = min(

        NUM_PREDICTIONS_TO_PLOT,

        images.size(
            0
        )

    )


    fig, axes = plt.subplots(

        n,

        3,

        figsize=(
            15,
            5 * n
        )

    )


    if n == 1:

        axes = np.expand_dims(

            axes,

            axis=0

        )


    for i in range(
        n
    ):

        # ====================================================
        # RADIOGRAFIA
        # ====================================================

        axes[
            i,
            0
        ].imshow(

            images[
                i,
                0
            ].numpy(),

            cmap="gray"

        )


        axes[
            i,
            0
        ].set_title(
            "Radiografia"
        )


        axes[
            i,
            0
        ].axis(
            "off"
        )


        # ====================================================
        # TARGET
        # ====================================================

        axes[
            i,
            1
        ].imshow(

            masks[
                i
            ].numpy(),

            vmin=0,

            vmax=(
                NUM_CLASSES - 1
            )

        )


        axes[
            i,
            1
        ].set_title(
            "Máscara verdadeira"
        )


        axes[
            i,
            1
        ].axis(
            "off"
        )


        # ====================================================
        # PREDIÇÃO
        # ====================================================

        axes[
            i,
            2
        ].imshow(

            predictions[
                i
            ].numpy(),

            vmin=0,

            vmax=(
                NUM_CLASSES - 1
            )

        )


        axes[
            i,
            2
        ].set_title(
            "Predição"
        )


        axes[
            i,
            2
        ].axis(
            "off"
        )


    fig.tight_layout()


    fig.savefig(

        OUTPUT_DIR
        /
        "test_predictions.png",

        dpi=160,

        bbox_inches="tight"

    )


    plt.close(
        fig
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # SEED
    # ========================================================

    set_seed(
        SEED
    )


    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else

        "cpu"

    )


    amp_enabled = (

        USE_AMP

        and

        device.type == "cuda"

    )


    if device.type == "cuda":

        torch.backends.cudnn.benchmark = (
            True
        )


    # ========================================================
    # CONFIGURAÇÃO
    # ========================================================

    print(
        "=" * 72
    )


    print(
        "U-NET + COORDENADAS X/Y "
        "+ FOCAL TVERSKY"
    )


    print(
        "=" * 72
    )


    print(

        f"PyTorch          : "
        f"{torch.__version__}"

    )


    print(

        f"Device           : "
        f"{device}"

    )


    if device.type == "cuda":

        print(

            f"GPU              : "
            f"{torch.cuda.get_device_name(0)}"

        )


    print(

        f"AMP              : "
        f"{amp_enabled}"

    )


    print(

        f"Resolução        : "
        f"{IMAGE_SIZE[0]}x"
        f"{IMAGE_SIZE[1]}"

    )


    print(

        f"Batch size       : "
        f"{BATCH_SIZE}"

    )


    print(

        f"Épocas máximas   : "
        f"{EPOCHS}"

    )


    print(

        f"Early stopping   : "
        f"{EARLY_STOPPING_PATIENCE}"

    )


    print(

        "Entrada U-Net    : "
        "imagem + X + Y"

    )


    print(

        f"Loss             : "
        f"{CE_WEIGHT:.2f} CE + "
        f"{FOCAL_TVERSKY_WEIGHT:.2f} "
        f"Focal Tversky"

    )


    print(

        f"Tversky          : "
        f"FP={TVERSKY_ALPHA_FP}, "
        f"FN={TVERSKY_BETA_FN}, "
        f"gamma={FOCAL_GAMMA}"

    )


    # ========================================================
    # JSONS
    # ========================================================

    json_files = sorted(

        ANNOTATIONS_DIR.glob(
            "*.json"
        )

    )


    if not json_files:

        raise RuntimeError(

            f"Nenhum JSON encontrado em: "
            f"{ANNOTATIONS_DIR.resolve()}"

        )


    # ========================================================
    # SPLIT
    # ========================================================

    (
        train_files,
        val_files,
        test_files

    ) = split_json_files(

        json_files,

        seed=SEED

    )


    print(
        "\n"
        +
        "=" * 72
    )


    print(
        "DIVISÃO DOS DADOS"
    )


    print(
        "=" * 72
    )


    print(

        f"Total            : "
        f"{len(json_files)}"

    )


    print(

        f"Treino           : "
        f"{len(train_files)}"

    )


    print(

        f"Validação        : "
        f"{len(val_files)}"

    )


    print(

        f"Teste            : "
        f"{len(test_files)}"

    )


    save_split_files(

        train_files,

        val_files,

        test_files

    )


    category_names = (
        load_category_names(

            json_files[0]

        )
    )


    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = (
        DentalPanoramicDataset(

            images_dir=(
                IMAGES_DIR
            ),

            json_files=(
                train_files
            ),

            image_size=(
                IMAGE_SIZE
            ),

            augment=True

        )
    )


    train_dataset_no_aug = (
        DentalPanoramicDataset(

            images_dir=(
                IMAGES_DIR
            ),

            json_files=(
                train_files
            ),

            image_size=(
                IMAGE_SIZE
            ),

            augment=False

        )
    )


    val_dataset = (
        DentalPanoramicDataset(

            images_dir=(
                IMAGES_DIR
            ),

            json_files=(
                val_files
            ),

            image_size=(
                IMAGE_SIZE
            ),

            augment=False

        )
    )


    test_dataset = (
        DentalPanoramicDataset(

            images_dir=(
                IMAGES_DIR
            ),

            json_files=(
                test_files
            ),

            image_size=(
                IMAGE_SIZE
            ),

            augment=False

        )
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = make_loader(

        train_dataset,

        shuffle=True

    )


    val_loader = make_loader(

        val_dataset,

        shuffle=False

    )


    test_loader = make_loader(

        test_dataset,

        shuffle=False

    )


    # ========================================================
    # PESOS
    # ========================================================

    if USE_CLASS_WEIGHTS:

        class_weights = (

            compute_class_weights(

                train_dataset_no_aug,

                NUM_CLASSES

            )

            .to(
                device
            )

        )

    else:

        class_weights = None


    # ========================================================
    # MODELO
    # ========================================================

    model = (

        UNetWithCoordinates(

            num_classes=(
                NUM_CLASSES
            ),

            base_channels=(
                BASE_CHANNELS
            )

        )

        .to(
            device
        )

    )


    total_parameters = sum(

        parameter.numel()

        for parameter
        in model.parameters()

    )


    trainable_parameters = sum(

        parameter.numel()

        for parameter
        in model.parameters()

        if parameter.requires_grad

    )


    print(
        "\n"
        +
        "=" * 72
    )


    print(
        "MODELO"
    )


    print(
        "=" * 72
    )


    print(

        f"Parâmetros totais     : "
        f"{total_parameters:,}"

    )


    print(

        f"Parâmetros treináveis : "
        f"{trainable_parameters:,}"

    )


    # ========================================================
    # TESTE DE SHAPE
    # ========================================================

    with torch.no_grad():

        dummy = torch.zeros(

            1,

            1,

            IMAGE_SIZE[0],

            IMAGE_SIZE[1],

            device=device

        )


        dummy_logits = model(
            dummy
        )


    print(

        f"Entrada externa       : "
        f"{tuple(dummy.shape)}"

    )


    print(

        f"Saída                 : "
        f"{tuple(dummy_logits.shape)}"

    )


    del dummy

    del dummy_logits


    if device.type == "cuda":

        torch.cuda.empty_cache()


    # ========================================================
    # LOSS
    # ========================================================

    criterion = CombinedLoss(

        num_classes=(
            NUM_CLASSES
        ),

        class_weights=(
            class_weights
        )

    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=(
            LEARNING_RATE
        ),

        weight_decay=(
            WEIGHT_DECAY
        )

    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = (

        torch
        .optim
        .lr_scheduler
        .ReduceLROnPlateau(

            optimizer,

            mode="min",

            factor=(
                SCHEDULER_FACTOR
            ),

            patience=(
                SCHEDULER_PATIENCE
            ),

            min_lr=(
                MIN_LR
            )

        )

    )


    # ========================================================
    # AMP
    # ========================================================

    scaler = torch.amp.GradScaler(

        device.type,

        enabled=(
            amp_enabled
        )

    )


    # ========================================================
    # VARIÁVEIS DE CONTROLE
    # ========================================================

    best_val_dice = (
        -math.inf
    )


    best_epoch = -1


    epochs_without_improvement = 0


    history = []


    checkpoint_path = (

        OUTPUT_DIR

        /

        "best_unet_coord_focal_tversky.pt"

    )


    start_time = time.time()


    # ========================================================
    # LOOP
    # ========================================================

    print(
        "\n"
        +
        "=" * 72
    )


    print(
        "TREINAMENTO"
    )


    print(
        "=" * 72
    )


    for epoch in range(

        1,

        EPOCHS + 1

    ):


        print(

            f"\nÉpoca "
            f"{epoch}/"
            f"{EPOCHS}"

        )


        # ====================================================
        # TREINO
        # ====================================================

        train_result = (
            train_one_epoch(

                model=model,

                loader=train_loader,

                criterion=criterion,

                optimizer=optimizer,

                scaler=scaler,

                device=device,

                amp_enabled=(
                    amp_enabled
                )

            )
        )


        # ====================================================
        # VALIDAÇÃO
        # ====================================================

        val_result = evaluate(

            model=model,

            loader=val_loader,

            criterion=criterion,

            device=device,

            amp_enabled=(
                amp_enabled
            ),

            description=(
                "Validação"
            )

        )


        # ====================================================
        # SCHEDULER
        # ====================================================

        scheduler.step(

            val_result[
                "loss"
            ]

        )


        current_lr = (

            optimizer
            .param_groups[0][
                "lr"
            ]

        )


        # ====================================================
        # HISTÓRICO
        # ====================================================

        row = {

            "epoch":
                epoch,

            "lr":
                current_lr,

            "train_loss":
                train_result[
                    "loss"
                ],

            "train_ce_loss":
                train_result[
                    "ce_loss"
                ],

            "train_focal_tversky_loss":
                train_result[
                    "focal_tversky_loss"
                ],

            "val_loss":
                val_result[
                    "loss"
                ],

            "val_ce_loss":
                val_result[
                    "ce_loss"
                ],

            "val_focal_tversky_loss":
                val_result[
                    "focal_tversky_loss"
                ],

            "val_pixel_accuracy":
                val_result[
                    "pixel_accuracy"
                ],

            "val_dice":
                val_result[
                    "dice_macro"
                ],

            "val_iou":
                val_result[
                    "iou_macro"
                ]

        }


        history.append(
            row
        )


        save_history(
            history
        )


        # ====================================================
        # PRINT
        # ====================================================

        print(

            f"Train Loss: "
            f"{train_result['loss']:.6f} | "

            f"Val Loss: "
            f"{val_result['loss']:.6f}"

        )


        print(

            f"Val Dice: "
            f"{val_result['dice_macro']:.6f} | "

            f"Val IoU: "
            f"{val_result['iou_macro']:.6f} | "

            f"Pixel Acc: "
            f"{val_result['pixel_accuracy']:.6f} | "

            f"LR: "
            f"{current_lr:.2e}"

        )


        # ====================================================
        # CHECKPOINT
        # ====================================================

        if (

            val_result[
                "dice_macro"
            ]

            >

            best_val_dice
            +
            1e-6

        ):

            best_val_dice = (

                val_result[
                    "dice_macro"
                ]

            )


            best_epoch = (
                epoch
            )


            epochs_without_improvement = 0


            torch.save(

                {

                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "best_val_dice":
                        best_val_dice,

                    "val_iou":
                        val_result[
                            "iou_macro"
                        ],

                    "config": {

                        "image_size":
                            IMAGE_SIZE,

                        "num_classes":
                            NUM_CLASSES,

                        "base_channels":
                            BASE_CHANNELS,

                        "epochs_max":
                            EPOCHS,

                        "coordinate_channels":
                            True,

                        "loss":
                            (
                                "CrossEntropy + "
                                "FocalTversky"
                            ),

                        "ce_weight":
                            CE_WEIGHT,

                        "focal_tversky_weight":
                            FOCAL_TVERSKY_WEIGHT,

                        "tversky_alpha_fp":
                            TVERSKY_ALPHA_FP,

                        "tversky_beta_fn":
                            TVERSKY_BETA_FN,

                        "focal_gamma":
                            FOCAL_GAMMA

                    }

                },

                checkpoint_path

            )


            print(
                "✓ Novo melhor modelo salvo."
            )


        else:

            epochs_without_improvement += 1


            print(

                "Sem melhora do Dice: "

                f"{epochs_without_improvement}/"

                f"{EARLY_STOPPING_PATIENCE}"

            )


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (

            epochs_without_improvement

            >=

            EARLY_STOPPING_PATIENCE

        ):

            print(
                "\nEarly stopping acionado."
            )

            break


    # ========================================================
    # TEMPO
    # ========================================================

    total_minutes = (

        time.time()

        -

        start_time

    ) / 60.0


    save_history(
        history
    )


    save_training_curves(
        history
    )


    # ========================================================
    # CARREGAR MELHOR MODELO
    # ========================================================

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device,

        weights_only=False

    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )


    # ========================================================
    # TESTE
    # ========================================================

    test_result = evaluate(

        model=model,

        loader=test_loader,

        criterion=criterion,

        device=device,

        amp_enabled=(
            amp_enabled
        ),

        description="Teste"

    )


    # ========================================================
    # SALVAR
    # ========================================================

    save_test_results(

        test_result,

        category_names

    )


    save_prediction_examples(

        model=model,

        loader=test_loader,

        device=device,

        amp_enabled=(
            amp_enabled
        )

    )


    # ========================================================
    # RESULTADO FINAL
    # ========================================================

    print(
        "\n"
        +
        "=" * 72
    )


    print(

        "RESULTADO FINAL — "
        "U-NET + COORDENADAS "
        "+ FOCAL TVERSKY"

    )


    print(
        "=" * 72
    )


    print(

        f"Melhor época          : "
        f"{best_epoch}"

    )


    print(

        f"Épocas executadas     : "
        f"{len(history)}"

    )


    print(

        f"Melhor Val Dice       : "
        f"{best_val_dice:.6f}"

    )


    print(

        f"Test Loss             : "
        f"{test_result['loss']:.6f}"

    )


    print(

        f"Test Dice macro       : "
        f"{test_result['dice_macro']:.6f}"

    )


    print(

        f"Test IoU macro        : "
        f"{test_result['iou_macro']:.6f}"

    )


    print(

        f"Test Pixel Accuracy   : "
        f"{test_result['pixel_accuracy']:.6f}"

    )


    print(

        f"Tempo total           : "
        f"{total_minutes:.2f} min"

    )


    print(

        f"Checkpoint            : "
        f"{checkpoint_path}"

    )


    print(

        f"Resultados            : "
        f"{OUTPUT_DIR.resolve()}"

    )


    # ========================================================
    # POR DENTE
    # ========================================================

    print(
        "\n"
        +
        "-" * 72
    )


    print(
        "MÉTRICAS POR CLASSE — TESTE"
    )


    print(
        "-" * 72
    )


    for class_id in range(

        1,

        NUM_CLASSES

    ):

        name = (
            category_names.get(

                class_id,

                str(class_id)

            )
        )


        dice = (

            test_result[
                "dice_per_class"
            ][class_id]

        )


        iou = (

            test_result[
                "iou_per_class"
            ][class_id]

        )


        support = int(

            test_result[
                "support_per_class"
            ][class_id]

        )


        print(

            f"Classe "
            f"{class_id:2d} "

            f"({name:>10}) | "

            f"Dice="
            f"{dice:.4f} | "

            f"IoU="
            f"{iou:.4f} | "

            f"pixels="
            f"{support}"

        )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()
