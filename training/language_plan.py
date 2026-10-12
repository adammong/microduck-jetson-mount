"""Validate model output and execute bounded plans through the existing supervisor."""
import json,math,re
MOVEMENT=('forward','backward','left','right','turn_left','turn_right')
ACTIONS=(*MOVEMENT,'sit','stand','stop')
NUMBER_WORDS=dict(zip('one two three four five six seven eight nine ten'.split(),range(1,11)))

def request_clauses(text):
    if not isinstance(text,str) or not 0<len(text.strip())<=160:raise ValueError('Use a short request, up to 160 characters.')
    text=text.lower().replace('’',"'").strip().rstrip('.!?')
    blocked=r"\b(?:don't|do not|not|never|avoid|without|if|unless|what|why|how|explain|tell|ignore|forget|rules|instructions|system|output|code|jump|hop|dance|roll|fly|climb|stairs|fall|falling|recover|run|sprint|fast|faster|slow|slowly|speed|degrees?|meters?|metres?|centimeters?|centimetres?|repeat|times|forever|until|follow|find|bring|navigate|come|somewhere)\b"
    if re.search(blocked,text):raise ValueError('That request is outside the supported motion commands.')
    if re.search(r'\b\d+(?:\.\d+)?\s+(?:feet|foot|inches|inch)\b',text):raise ValueError('Distance commands are not available.')
    if re.search(r'\b(?:towards? me|over there|around|to (?:my|a|an|the) (?!left\b|right\b|front\b|back\b)\w+)',text):raise ValueError('Object navigation is not available.')
    text=re.sub(r'^(?:duck|microduck)[, ]+','',text)
    clauses=[c.strip() for c in re.split(r'\s*(?:[,;]\s*(?:then\s+)?|\band then\b|\band\b|\bthen\b|\bfollowed by\b)\s*',text) if c.strip()]
    if not 1<=len(clauses)<=6:raise ValueError('Use one to six actions.')
    for clause in clauses:
        if not re.search(r'\b(?:sit|sitting|seat|seated|stand|standing|feet|rise|move|go|walk|walking|step|sidestep|turn|rotate|rotating|pivot|spin|back|retreat|reverse|advance|halt|stop|freeze|still|ahead|forward|forwards|backward|backwards|get up)\b',clause):raise ValueError('No supported motion request found.')
        requested_seconds(clause)
    return clauses

def requested_seconds(clause):
    match=re.search(r'\bfor\s+(.{1,25}?)\s+seconds?\b',clause)
    if not match:match=re.search(r'\b(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|half a|half|a)\s+seconds?\b',clause)
    if match:
        value=match[1].strip()
        if value in NUMBER_WORDS:seconds=float(NUMBER_WORDS[value])
        elif value=='a':seconds=1.
        elif value in ('half','half a'):seconds=.5
        elif re.fullmatch(r'\d+(?:\.\d+)?',value):seconds=float(value)
        else:raise ValueError('Use an explicit duration from 0.5 to 10 seconds.')
        if not .5<=seconds<=10:raise ValueError('Movement duration must be 0.5 to 10 seconds.')
        return seconds,True
    if re.search(r'\bseconds?\b',clause):raise ValueError('Use an explicit duration from 0.5 to 10 seconds.')
    return (1. if re.search(r'\b(?:a little|a bit|briefly)\b',clause) else 3.),False

def validate_request_plan(text,steps):
    clauses=request_clauses(text);steps=validate_plan(dict(steps=steps))
    if len(steps)!=len(clauses):raise ValueError('The model missed or added an action. Please rephrase.')
    for clause,step in zip(clauses,steps):
        # The model may invent a valid action; explicit words must agree too.
        posture=False
        for pattern,expected in [(r'\b(?:sit|sitting|seat|seated)\b','sit'),(r'\b(?:stand|standing|feet|rise|get up)\b','stand'),(r'\b(?:stop|halt|freeze|still)\b','stop')]:
            if re.search(pattern,clause) and step['action']!=expected:raise ValueError('The model changed the requested action. Please rephrase.')
            if re.search(pattern,clause):posture=True
        moving=re.search(r'\b(?:move|walk|step|sidestep|turn|rotate|rotating|pivot|spin|back|retreat|reverse|advance|ahead|forwards?|backwards?)\b',clause)
        if moving and not posture and step['action'] not in MOVEMENT:raise ValueError('The model changed movement to a posture. Please rephrase.')
        rotating=bool(re.search(r'\b(?:turn|rotate|rotating|pivot|spin)\b',clause))
        for pattern,direction in [(r'\bleft\b|\bcounterclockwise\b','left'),(r'\bright\b|\bclockwise\b','right'),(r'\b(?:forward|forwards|ahead|advance)\b','forward'),(r'\b(?:backward|backwards|back|reverse|retreat)\b','backward')]:
            if not posture and re.search(pattern,clause) and step['action']!=('turn_'+direction if rotating and direction in ('left','right') else direction):raise ValueError('The model changed the requested direction. Please rephrase.')
        seconds,explicit=requested_seconds(clause)
        if step['action'] in MOVEMENT:
            if step['seconds']!=seconds:raise ValueError('The model changed the requested duration. Please rephrase.')
        elif explicit:raise ValueError('Durations apply only to movement.')
    return steps

def validate_plan(value):
    if isinstance(value,str):value=json.loads(value)
    if not isinstance(value,dict) or set(value)!={'steps'}:raise ValueError('Expected a motion plan.')
    steps=value['steps']
    if not isinstance(steps,list) or not 1<=len(steps)<=6:raise ValueError('Use one to six supported actions.')
    total=0.;clean=[]
    for step in steps:
        if not isinstance(step,dict) or not isinstance(step.get('action'),str) or step['action'] not in ACTIONS:raise ValueError('Unsupported action.')
        action=step['action']
        if action in MOVEMENT:
            if set(step)!={'action','seconds'}:raise ValueError('Movement needs a duration.')
            seconds=step['seconds']
            if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or not .5<=seconds<=10:raise ValueError('Movement duration must be 0.5 to 10 seconds.')
            total+=seconds;clean.append(dict(action=action,seconds=float(seconds)))
        else:
            if set(step)!={'action'}:raise ValueError('Posture and stop do not take a duration.')
            clean.append(dict(action=action))
    if total>20:raise ValueError('A plan can contain at most 20 seconds of movement.')
    if any(s['action']=='stop' for s in clean[:-1]):raise ValueError('Stop must be the final action.')
    return clean

def intent_for(step):
    from skill_controller import parse_command
    text=step['action'].replace('_',' ')
    if step['action'] in MOVEMENT:text+=f" for {step['seconds']} seconds"
    return parse_command(text)

class PlanRunner:
    def __init__(self):self.cancel()
    def cancel(self):self.remaining=[];self.current=None;self.started=0.;self.stable=0.
    def start(self,controller,steps,now):
        steps=validate_plan(dict(steps=steps))
        controller.accept(intent_for(steps[0]),now)
        self.remaining=steps[1:];self.current=steps[0];self.started=now;self.stable=0.
    def update(self,controller,measurement,now,dt=.02):
        if not self.current:return
        if controller.fault:self.cancel();return
        if now-controller.heartbeat_at>.6:
            controller.accept(intent_for(dict(action='stop')),now);self.cancel();return
        if now-self.started>25:
            controller.fail('Plan action timed out',now);self.cancel();return
        action=self.current['action']
        ready=controller.phase=='seated' and measurement['sitting'] if action=='sit' else controller.phase=='idle' and measurement['standing']
        if action=='stop' and controller.phase=='seated':ready=measurement['sitting']
        if controller.phase=='idle':ready=ready and measurement['motion_settled']
        ready=ready and controller.pending is None and now-self.started>=.5
        self.stable=self.stable+dt if ready else 0.
        if self.stable>=.8:
            if self.remaining:
                step=self.remaining.pop(0);controller.accept(intent_for(step),now)
                self.current=step;self.started=now;self.stable=0.
            else:self.cancel()
    def status(self):return dict(current=self.current,remaining=list(self.remaining))
