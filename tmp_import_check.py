import importlib
import traceback

modules = [
	'cli.run',
	'core.knowledge_base.base',
	'core.knowledge_base.main_kb',
	'core.knowledge_base.temp_kb',
	'core.knowledge_base.kg_loader',
	'core.llm.base',
	'core.llm.service',
	'core.llm.models',
]

for m in modules:
	print(f'>> Import {m} ...')
	try:
		mod = importlib.import_module(m)
		print(f'OK: {getattr(mod, "__file__", None)}')
	except Exception:
		print(f'FAIL: {m}')
		traceback.print_exc()
	print('-'*60)
