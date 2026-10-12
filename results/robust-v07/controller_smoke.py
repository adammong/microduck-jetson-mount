from pathlib import Path
p=Path('results/terrain-v07/controller_smoke.py')
s=p.read_text().replace('tmp/terrain-study','tmp/robust-study').replace('results/terrain-v07','results/robust-v07').replace("'1701'","'1601'").replace("'711'","'601'")
exec(compile(s,'controller_smoke.py','exec'))
