"""
grl.py

Gradient Reversal Layer (GRL), il componente chiave di DANN (Domain-
Adversarial Neural Network, Ganin & Lempitsky 2015).

Nel forward pass si comporta come l'identita' (non modifica i dati).
Nel backward pass, INVERTE il segno del gradiente (moltiplicato per un
coefficiente lambda) prima che raggiunga il feature extractor.

Effetto pratico: il domain discriminator viene allenato normalmente a
distinguere source da target, mentre il feature extractor - proprio
perche' riceve il gradiente invertito - viene spinto nella direzione
OPPOSTA, cioe' a produrre feature che il discriminator non riesce a
distinguere. Questo permette di allenare tutto con un singolo backward
pass e un singolo optimizer, senza dover alternare manualmente due fasi
di training (min-max) come si farebbe in una GAN classica.
"""

from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)
