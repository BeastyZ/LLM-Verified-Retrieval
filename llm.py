import os
from transformers import AutoTokenizer
from utils import *
import openai
class LLM:

    def __init__(self, args):
        self.args = args

        if args.openai_api:
            # logger.info('into if args.openai_api:')
            import openai
            OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
            OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")
            OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
            assert OPENAI_API_KEY == None, "api_key={}".format(OPENAI_API_KEY)

            if args.azure:

                openai.api_key = OPENAI_API_KEY
                openai.api_base = OPENAI_API_BASE
                openai.api_type = 'azure'
                openai.api_version = '2022-12-01'
            else:
                # logger.info('into not args.azure')
                # logger.info('OPENAI_API_KEY:{}'.format(OPENAI_API_KEY))
                openai.api_key = OPENAI_API_KEY
                openai.organization = OPENAI_ORG_ID

            self.tokenizer = AutoTokenizer.from_pretrained("gpt2",
                                                           fast_tokenizer=False)  # TODO: For ChatGPT we should use a different one
            self.total_tokens = 0  # To keep track of how much the API costs
        else:
            self.model, self.tokenizer = load_model(args.model)

        self.prompt_exceed_max_length = 0
        self.fewer_than_50 = 0
        self.azure_filter_fail = 0

    def generate(self, prompt, max_tokens, api_key, stop=None, turbo_system_message=None):
        args = self.args
        if max_tokens == 0:
            self.prompt_exceed_max_length += 1
            logger.warning(
                "Prompt exceeds max length and return an empty string as answer. If this happens too many times, it is suggested to make the prompt shorter")
            return ""
        if max_tokens < 50:
            self.fewer_than_50 += 1
            logger.warning(
                "The model can at most generate < 50 tokens. If this happens too many times, it is suggested to make the prompt shorter")

        if args.openai_api:
            if "turbo" in args.model and not args.azure:
                assert turbo_system_message != None
                # For OpenAI's ChatGPT API, we need to convert text prompt to chat prompt
                prompt = [
                    # {'role': 'system', 'content': "You are a helpful assistant that answers the following questions with proper citations."},
                    {'role': 'system', 'content': turbo_system_message},
                    {'role': 'user', 'content': prompt}
                ]
            else:
                if "turbo" in args.model:
                    deploy_name = "gpt-35-turbo-0301"
                else:
                    deploy_name = args.model

            def repeat_until_success_call_openai_api_only_for_retry(func):
                def wrapper(*args, **kw):
                    while 1:
                        result = None
                        try:
                            result = func(*args, **kw)
                        except openai.error.APIConnectionError as e:
                            logger.warning('openai connection error, so retry after sleep 1 seconds')
                            logger.warning(e)
                            time.sleep(1)
                        except openai.error.RateLimitError as e:
                            logger.warning(type(e))
                            if 'quota' in e._message:
                                raise e
                            else:
                                time.sleep(60)
                        except openai.error.AuthenticationError as e:
                            raise e
                        except Exception as e:
                            logger.warning('meet unexpected error, so retry after sleep 3 seconds')
                            logger.warning(e)
                            logger.warning(type(e))
                            time.sleep(3)

                        if result != None:
                            return result
                        else:
                            pass
                return wrapper

            if "turbo" in args.model and not args.azure:
                @repeat_until_success_call_openai_api_only_for_retry
                def tmp_openai_call_func():
                    response = openai.ChatCompletion.create(
                        model=args.model,
                        messages=prompt,
                        temperature=args.temperature,
                        max_tokens=max_tokens,
                        stop=stop,
                        top_p=args.top_p,
                        api_key=api_key,
                        n=self.args.num_samples,
                    )
                    return response
                response = tmp_openai_call_func()
                self.total_tokens += response['usage']['total_tokens']
                result = list(map(lambda x:x['message']['content'],response['choices']))
                return result
            else:

                @repeat_until_success_call_openai_api_only_for_retry
                def tmp_openai_call_func():
                    response = openai.ChatCompletion.create(
                        model=args.model,
                        messages=prompt,
                        temperature=args.temperature,
                        max_tokens=max_tokens,
                        stop=stop,
                        top_p=args.top_p,
                        api_key=api_key,
                        n=self.args.num_samples
                    )
                    return response
                response = tmp_openai_call_func()
                self.total_tokens += response['usage']['total_tokens']
                result = list(map(lambda x:x['text'],response['choices']))
                return result
        else:

            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
            stop = [] if stop is None else stop
            stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"]))  # In Llama \n is <0x0A>; In OPT \n is Ċ
            stop_token_ids = list(set([self.tokenizer._convert_token_to_id(stop_token) for stop_token in stop] + [
                self.model.config.eos_token_id]))
            if "llama" in args.model:
                stop_token_ids.remove(self.tokenizer.unk_token_id)
            outputs = self.model.generate(
                **inputs,
                do_sample=True, temperature=args.temperature, top_p=args.top_p,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                eos_token_id=stop_token_ids
            )
            generation = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
            return generation