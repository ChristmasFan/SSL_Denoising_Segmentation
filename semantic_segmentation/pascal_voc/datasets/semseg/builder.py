import logging

#from .cityscapes import build_cityloader
from datasets.semseg.pascal_voc import build_vocloader
#from .cityscapes_semi_cp import build_city_semi_loader_cp
from datasets.semseg.pascal_voc_semi_cp import build_voc_semi_loader_cp

logger = logging.getLogger('global')


def get_loader(cfg, seed=0):
    cfg_dataset = cfg['dataset']

    #if cfg_dataset['type'] == 'cityscapes_semi_cp':
    #    trainloader_sup, trainloader_unsup = build_city_semi_loader_cp('train', cfg, seed=seed)
    #    valloader = build_cityloader('val', cfg)
    if cfg_dataset['type'] == 'pascal_voc_semi_cp':
        #trainloader_sup, trainloader_unsup = build_voc_semi_loader_cp('train',cfg, seed=seed)
        valloader = build_vocloader('val',cfg)
    else:
        raise NotImplementedError("dataset type {} is not supported".format(cfg_dataset))
    logger.info('Get loader Done...')

    return trainloader_sup, trainloader_unsup, valloader

def main():
    config = {}
    config["dataset"] = {}
    config["dataset"]["type"] = "pascal_voc_semi_cp"
    config["dataset"]["mean"] = [123.675, 116.28, 103.53]
    config["dataset"]["std"] = [58.395, 57.12, 57.375]
    config["dataset"]["ignore_label"] = 255
    get_loader(config)


if __name__ == "__main__":
    main()