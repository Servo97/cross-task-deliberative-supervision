"""Fail-closed GROOT dataset bases for isolated RoboMME training."""

LE_ROBOT_MODALITY_FILENAME = "modality.json"


class _UnavailableRoboCasaDataset:
    def __init__(self, *_args, **_kwargs):
        raise RuntimeError(
            "RoboCasa/GROOT datasets are unavailable in an isolated RoboMME job; "
            "the dedicated RoboMME LeRobot adapter must be selected"
        )


class LeRobotMixtureDataset(_UnavailableRoboCasaDataset):
    pass


class LeRobotSingleDataset(_UnavailableRoboCasaDataset):
    pass


class ModalityConfig(_UnavailableRoboCasaDataset):
    pass
