import os
import datetime
import wandb
from dotenv import load_dotenv


class Wandb:
    def __init__(self, run_id, **kwargs):
        """
        wandb args의 auth key값을 사용하여 로그인
        """
        self.args = kwargs
        self.run_id = run_id
        
        self.entity = self.args["wandb_entity"]
        self.project = self.args["wandb_project"]
        dotenv_path = self.args["wandb_env_path"]

        load_dotenv(dotenv_path=dotenv_path)
        WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
        wandb.login(key=WANDB_AUTH_KEY)

    def init_wandb(self, **kwargs):
        """
        :param args: arguments변수
        :param **kwargs: wandb의 태그와 name에 추가하고싶은 내용을 넣어줌 ex_ fold=1
        """
        data_dirs = self.args['data_dir']
        if isinstance(data_dirs, list):
            data_dirs = ' '.join(data_dirs)

        name = f"{data_dirs}_{self.args['train_transform']}_{self.run_id}"
        tags = []

        if self.args['wandb_unique_tag']:
            tags = [f"wandb_unique_tag: {self.args['wandb_unique_tag']}"]
            name += f"_{self['wandb_unique_tag']}"

        for k, v in self.args.items():
            tags.append(f"{k}: {v}")
        wandb.init(config=self.args, tags=tags, entity=self.entity, project=self.project, reinit=True)
        wandb.run.name = name

    def log(self, phase, **kwargs):
        """
        wandb에 차트 그래프를 그리기 위해 로그를 찍는 함수
        :param phase: 'train' or 'valid'
        :param **kwargs: {'loss': 4.937147756417592, 'Cls loss': 0.8772911429405212, 'Angle loss': 2.277783346672853, 'IoU loss': 1.7820731500784557}
        """
        new_log = {}
        for metric in kwargs.keys():
            key = f"{phase}/{metric}"
            new_log[key] = kwargs[metric]
        wandb.log(new_log)