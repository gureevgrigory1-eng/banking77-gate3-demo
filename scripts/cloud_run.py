"""Run build/eval in authenticated Cloud Shell; never emit or persist secrets."""
import json,os,subprocess,sys,runpy
payload=json.loads(subprocess.check_output(['yc','lockbox','payload','get','--id','e6qit92oanvr0bbuasrn','--version-id','e6q6ov26fe3iubmj49oq','--format','json']))
entry=next(e for e in payload['entries'] if e['key']=='GIGACHAT_AUTH_KEY')
os.environ['GIGACHAT_AUTH_KEY']=entry['text_value'] if 'text_value' in entry else entry['textValue']
del payload,entry
try:runpy.run_path(sys.argv[1],run_name='__main__')
finally:os.environ.pop('GIGACHAT_AUTH_KEY',None)
