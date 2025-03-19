from datetime import datetime
import argparse
import os
import csv

BASE_FOLDER = './metrics'
METRICS_HEADER = ['Epoch', 'Loss', 'Elapsed Time']

class CSV_Metric:
    def __init__(self, args: argparse.Namespace):
        self.filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.encoder}.csv"
        self.t1 = None
        self.t2 = None    
        self.args = args
        self._make_file()
        
    def _make_file(self) -> str:
        if not os.path.exists(BASE_FOLDER):
            os.makedirs(BASE_FOLDER)

        if not os.path.exists(f"{BASE_FOLDER}/{self.args.dataset_name}"):
            os.makedirs(f"{BASE_FOLDER}/{self.args.dataset_name}")
            
        if not os.path.exists(f"{BASE_FOLDER}/{self.args.dataset_name}/{self.filename}"):
            with open(f"{BASE_FOLDER}/{self.args.dataset_name}/{self.filename}", "w", newline="") as file:
                file.write(f"#{vars(self.args)}\n")
                writer = csv.writer(file)
                writer.writerow(METRICS_HEADER)
                
    def start(self):
        self.t1 = datetime.now()
        return
        
    def end(self):
        self.t2 = datetime.now()
        return
        
    def write(self, epoch: int, loss: float):
        with open(f"{BASE_FOLDER}/{self.args.dataset_name}/{self.filename}", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, loss, (self.t2-self.t1).seconds])
        return