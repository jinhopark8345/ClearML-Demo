
# To find where the logging happens


### RLlib -> log with TensorboardX by default

- result = algo.train() ->
- self.log_result(result) ->
- self._result_logger.on_result(result) ->
- TBXLogger, on_result(result) ->
- flat_result and use _file_writer (_file_writer is the SummaryWriter from tensorboardX)

### ClearML pick up TensorboardX logging automatically
