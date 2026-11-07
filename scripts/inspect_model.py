"""Print the model architecture and trainable parameter inventory."""

from __future__ import annotations

from cadica.models import CADICACNN


def main() -> None:
    model = CADICACNN()

    print("MODEL ARCHITECTURE")
    print(model)

    print("\nTRAINABLE PARAMETERS")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(name, tuple(parameter.shape))

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print("\nTotal trainable parameters:", total)


if __name__ == "__main__":
    main()
