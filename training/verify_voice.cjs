/* Check browser PCM resampling without accessing a microphone. */
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync('training/skill_control.html','utf8');
const source=html.slice(html.indexOf('function wav('),html.indexOf('\nvoice.onclick='));
const context=vm.createContext({Float32Array,ArrayBuffer,DataView,Blob,Math});vm.runInContext(source,context);
(async()=>{
 for(const rate of [16000,44100,48000]){
  const input=new Float32Array(rate);for(let i=0;i<rate;i++)input[i]=2*Math.sin(2*Math.PI*440*i/rate);
  const blob=context.wav([input.slice(0,10000),input.slice(10000)],rate),buffer=await blob.arrayBuffer(),v=new DataView(buffer);
  assert.equal(buffer.byteLength,32044);assert.equal(v.getUint32(24,true),16000);assert.equal(v.getUint16(22,true),1);assert.equal(v.getUint16(34,true),16);
  assert.equal(v.getUint32(40,true),32000);assert.equal(Buffer.from(buffer).subarray(0,4).toString(),'RIFF');
  for(let i=0;i<16000;i++)assert.ok(Math.abs(v.getInt16(44+2*i,true))<=32767);
 }
 const output={passed:true,input_rates_hz:[16000,44100,48000],output_rate_hz:16000,mono_pcm16:true,microphone_recording_tested:false};
 fs.mkdirSync('results/voice-v07',{recursive:true});fs.writeFileSync('results/voice-v07/browser-audio-verification.json',JSON.stringify(output,null,2)+'\n');console.log('Browser WAV/resampling checks passed');
})();
