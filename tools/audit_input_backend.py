"""Same BF16 inputs as calibration, with explicit native grounding coordinates."""
from model_backends import finite_logits_processors
from protocol_v2_backend import QwenV2Backend,processor_inputs
from run_processed_grounding import PROMPT


class AuditBackend(QwenV2Backend):
    def generate(self,item):
        torch=self.torch
        inputs,usage=processor_inputs(self.processor,item)
        if item['axis']=='grounding':
            grid=usage['image_grid_thw']
            if len(grid)!=1 or grid[0][0]!=1: raise ValueError('Grounding requires one static image')
            patch=self.processor.image_processor.patch_size
            height,width=int(grid[0][1]*patch),int(grid[0][2]*patch)
            actual_prompt=PROMPT.format(expression=item['referring_expression'],width=width,height=height)
            inputs,usage=processor_inputs(self.processor,{**item,'question':actual_prompt})
            if usage['image_grid_thw']!=grid: raise ValueError('Grounding rendering changed image geometry')
            usage.update(processed_width=width,processed_height=height,actual_prompt=actual_prompt)
        inputs={k:v.to('cuda',dtype=torch.bfloat16) if v.is_floating_point() else v.to('cuda') for k,v in inputs.items()}
        with torch.inference_mode():
            output=self.model.generate(**inputs,max_new_tokens=128,do_sample=False,num_beams=1,
                                       logits_processor=finite_logits_processors())
        usage['output_tokens']=int(output.shape[1]-inputs['input_ids'].shape[1])
        eos=self.model.generation_config.eos_token_id
        eos=[] if eos is None else [eos] if isinstance(eos,int) else list(eos)
        usage['ended_with_eos']=int(output[0,-1]) in eos
        usage['truncated_at_output_cap']=usage['output_tokens']>=128 and not usage['ended_with_eos']
        self.last_usage=usage
        return self.processor.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
