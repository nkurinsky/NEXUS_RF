import sys, os
import subprocess
import numpy as np

source_topleveldir = '/data'
target_topleveldir = '/data-backup'


search_subdirs = (
    'USRP_Background_Data',
    # 'USRP_Laser_Data',
    # 'USRP_Laser_TempScan_Data',
    # 'USRP_Noise_Scans',
    # 'USRP_Source_Data',
)

## Get the list of source and target directories
source_dirs = [os.path.join(source_topleveldir,srcdir) for srcdir in search_subdirs]
target_dirs = [os.path.join(target_topleveldir,srcdir) for srcdir in search_subdirs]


if __name__ == "__main__":

    # ## Do this until eternity
    # while(1):

    ## Check each source directory
    for srcdir, tgtdir in zip(source_dirs,target_dirs):

        ## Get the list of dates in this file
        src_datedirs = np.sort(os.listdir(srcdir))

        ## Loop over all the date directories
        for datedir in src_datedirs:

            src_datedir = os.path.join(srcdir,datedir)
            tgt_datedir = os.path.join(tgtdir,datedir)

            ## Get the list of series directories in the source date directory
            src_serieslist = np.sort(os.listdir(src_datedir))

            ## Check if this date exists in the backup/target
            ## if not, create it (provided there is data to be copied)
            if (len(src_serieslist)>0) and (not os.path.isdir(tgt_datedir)):
                print("Creating directory:", tgt_datedir)
                # os.mkdir(tgt_datedir)
            else: print("Skipping creation of directory:", tgt_datedir, "as it already exists.")

            for seriesdir in src_serieslist:

                ## Get the full path for the source directory of interest, and the files 
                ## that are there
                src_seriesdir = os.path.join(src_datedir,seriesdir)
                src_allfiles  = os.listdir(src_seriesdir)

                tgt_filename  = os.path.join(tgt_datedir,seriesdir+".tar.gz")
                
                ## Check for a completed acquisition that hasn't already been copied
                if ('acq-complete' in src_allfiles): #and not os.exists(tgt_filename):

                    if not os.path.exists(tgt_filename):

                        ## If the targz does not exist yet:
                        ## Create a tarball of current src directory, place in tgt directory
                        cwd = os.getcwd() ; os.chdir(src_seriesdir)
                        tar_cmd = "tar -czf " + tgt_filename + " ./*.h5"
                        print("Calling command:", tar_cmd)
                        # return_code = subprocess.run(tar_cmd.split(" "), shell=True)
                        
                        ## Run the eventerizer on all files, then delete them
                        for srcfile in src_allfiles:

                            ## Only run eventerizer on the right files
                            if (srcfile[-3:]==".h5") and ("USRP_" in srcfile.split('/')[-1]):

                                if ("VNA" in srcfile.split('/')[-1]) or ("Noise" in srcfile.split('/')[-1]):
                                    continue

                                ## Run the eventerizer on this file
                                print("Running eventerizer on:", srcfile)

                                ## Delete this source file to keep disk space available
                                print("Deleting:", srcfile)

                    else:
                        print("Skipping series:", tgt_filename, "as it has already been copied.")

                ## If the acquisition hasn't been completed, or the data has already been copied
                else:
                    print("Skipping series:", tgt_filename, "as it is not finished.") 
                    continue


                # tgt_seriesdir = os.path.join(tgt_datedir,seriesdir)

