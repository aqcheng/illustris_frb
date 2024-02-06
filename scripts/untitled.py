argp = argparse.ArgumentParser()
argp.add_argument("-s", "--sim", type=str, required=True, choices=os.listdir('/home/tnguser/sims.TNG'), 
                  help="Name of simulation as given in the path, e.g. L205n2500TNG")
argp.add_argument("--binsize", type=int, default=500, help="The size of a bin in ckpc/h. Default=500")
argp.add_argument("--snaps", type=int, default=[99], nargs='+', help="Which snapshots to process. Default=99")
argp.add_argument("--snap-range", type=int, nargs=2, help="Range of snapshots to process, inclusive. Ignores --snaps if specified. ")
argp.add_argument("--outpath", type=str, default=None, help="Path to where the output electron density map will go. If unspecified, will go to ./n_e_maps/{sim}")