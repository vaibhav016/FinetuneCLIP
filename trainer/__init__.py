
from trainer.finetune import FinetuneCLIP, FinetuneFFN, FinetuenProj, FinetuneTextProj, FinetuenProjTV
from trainer.frozenclip import FrozenCLIP
from trainer.spu import MASEDIT
METHOD = {'FrozenCLIP': FrozenCLIP,
          'Finetune': FinetuneCLIP,
          'finetunevisual': FinetuneCLIP,
          'FinetuneFFN': FinetuneFFN,
          'FinetuneCproj': FinetuenProj,
          'FinetuneCprojboth': FinetuenProjTV,
          'FinetuneTextCproj': FinetuneTextProj,
          'SPU': MASEDIT,
          }
