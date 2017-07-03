use like this :
```bash
./plot_phase.py path/to/simulation/P_Opsi.dat \
                -s path/to/simulation/res.png \
                -o path/to/simulation/res.json \
                --radius 5.0 --width 1.2 \
                --delta_y 0.07 --delta_z 0.07 \
                --number_samples_y 280 \
                --number_samples_z 450 \
                --snap 1000
```

* `res.png` will contain the phase curves
* `res.json` will contain the winding numbers so that they can be easily loaded and analysed with any language later