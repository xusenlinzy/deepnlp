import random
from .trainer_base import TrainerBase
from ..callback import ProgressBar


class ExtractionTrainer(TrainerBase):
    """ 抽取类任务
    """
    
    keys_to_ignore_on_gpu = ['offset_mapping', 'target', 'texts', 'spn_labels', 'entity_types']  # batch不存放在gpu中的变量

    def build_batch_inputs(self, batch):
        """
        Sent all model inputs to the appropriate device (GPU on CPU)
        return:
         The inputs are in a dictionary format
        """
        inputs = {key: (
            value.to(self.device) if (
                    (key not in self.keys_to_ignore_on_gpu) and (value is not None)
            ) else value
        ) for key, value in batch.items()
        }

        if 'spn_labels' in batch.keys():
            inputs['spn_labels'] = [{key: value.to(self.device) for key, value in t.items()} for t in
                                    batch['spn_labels']]
        return inputs

    def build_batch_concat(self, all_batch_list, dim=0):
        preds, target = [], []
        for batch in all_batch_list:
            preds.extend(batch['predictions'])
            target.extend(batch['groundtruths'])

        for index in random.sample(range(len(preds)), 1):
            self.logger.info(f"Ground truth of the {index}th sentence is : \n {target[index]}")
            self.logger.info(f"Prediction of the {index}th sentence is : \n {preds[index]}")
        return {"preds": preds, "target": target}

    def predict(self, test_data, save_result=True, file_name=None, save_dir=None):
        """
        test数据集预测
        """
        all_batch_list = []
        test_dataloader = self.build_test_dataloader(test_data)
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Predicting')
        for step, batch in enumerate(test_dataloader):
            batch = self.predict_forward(batch)
            all_batch_list.extend(batch['predictions'])
            pbar.step(step)

        if save_result:
            if file_name is None:
                file_name = "test_predict_results.json"
            self.save_predict_result(data=all_batch_list, file_name=file_name, save_dir=save_dir)
