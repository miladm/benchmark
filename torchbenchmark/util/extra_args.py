import argparse
from typing import List

from torchbenchmark.util.backends.fx2trt import enable_fx2trt

# Dispatch arguments based on model type
def parse_args(model: 'torchbenchmark.util.model.BenchmarkModel', extra_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fx2trt", action='store_true', help="enable fx2trt")
    # by default, enable half precision for inference
    args = parser.parse_args(extra_args)
    args.device = model.device
    args.jit = model.jit
    # some models don't support train or eval tests, therefore they don't have these attributes
    args.train_bs = model.train_bs if hasattr(model, 'train_bs') else None
    args.eval_bs = model.eval_bs if hasattr(model, 'eval_bs') else None
    return args

def apply_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace):
   if args.fx2trt:
       assert args.device == 'cuda', "fx2trt is only available with CUDA."
       assert args.eval_bs, "fx2trt only applies to CUDA eval tests."
       assert not args.jit, "fx2trt with JIT is not available."
       model.eval_model = enable_fx2trt(args.eval_bs, args.eval_fp16, model.eval_model, model.eval_example_inputs)
