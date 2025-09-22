import subprocess

my_jobs = subprocess.check_output(["squeue", '--me', '--format="%.5i %.10M"']).decode().split('\n')


time = my_jobs[1].split(' ')[-1][:-1]
seconds = 0
for i,x in enumerate(time.split(':')[::-1]):
    seconds += int(x) * 60**i

lines = int(subprocess.check_output(["wc", "-l", "/climca/people/ppfleiderer/decomposition/ridge_out/TREFHT_6m7m8_stream_vX_alpha1_1500_1979-2023/train1500_test1300_trend_circ.csv"]).decode().split(' ')[0])
print(time, seconds, lines, lines / seconds)


quit()



time = my_jobs[1].split(' ')[-1][:-1]
seconds = 0
for i,x in enumerate(time.split(':')[::-1]):
    seconds += int(x) * 60**i
lines = int(subprocess.check_output(["wc", "-l", "/climca/people/ppfleiderer/decomposition/ridge_out/TREFHT_6m7m8_stream_vX_alpha1_1400_1979-2023/train1400_test1500_trend_circ.csv"]).decode().split(' ')[0])
print(time, seconds, lines, lines / seconds)


