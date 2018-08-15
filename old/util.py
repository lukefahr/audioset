
import logging 
import sys

import clipFind

#
#
#
#
#
if __name__ == '__main__':
    
    logging.basicConfig ( level = logging.WARN,
                        format='%(levelname)s [0x%(process)x] %(name)s: %(message)s')
     
    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == 'names':
            assert( len(sys.argv) > 2)

            ytid = sys.argv[2]

            cf= clipFind.ClipFinder( audioset='eval,balanced,unbalanced', logLvl=logging.DEBUG)
            
            names = cf.nameLookup(ytid)

            print ('Found Names: ' + ', '.join(names) )
            


