from my_datasets import CIFAR10


class TestCIFAR10:
    def test_instantiation(self):
        batch_size = 4
        cifar10 = CIFAR10(batch_size=batch_size)
        assert cifar10.batch_size == 4

    def test_get_trainloader(self):
        batch_size = 4
        cifar10 = CIFAR10(batch_size=batch_size)
        trainloader = cifar10.get_trainloader()
        assert len(trainloader) > 10
