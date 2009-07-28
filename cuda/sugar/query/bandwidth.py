#!/usr/bin/env python

MEMCOPY_ITERATIONS =  10
DEFAULT_SIZE       = ( 32 * ( 1 << 20 ) )    #32 M
DEFAULT_INCREMENT  = (1 << 22)               #4 M
CACHE_CLEAR_SIZE   = (1 << 24)               #16 M

#shmoo mode defines
SHMOO_MEMSIZE_MAX     = (1 << 26)         #64 M
SHMOO_MEMSIZE_START   = (1 << 10)         #1 KB
SHMOO_INCREMENT_1KB   = (1 << 10)         #1 KB
SHMOO_INCREMENT_2KB   = (1 << 11)         #2 KB
SHMOO_INCREMENT_10KB  = (10 * (1 << 10))  #10KB
SHMOO_INCREMENT_100KB = (100 * (1 << 10)) #100 KB
SHMOO_INCREMENT_1MB   = (1 << 20)         #1 MB
SHMOO_INCREMENT_2MB   = (1 << 21)         #2 MB
SHMOO_INCREMENT_4MB   = (1 << 22)         #4 MB
SHMOO_LIMIT_20KB      = (20 * (1 << 10))  #20 KB
SHMOO_LIMIT_50KB      = (50 * (1 << 10))  #50 KB
SHMOO_LIMIT_100KB     = (100 * (1 << 10)) #100 KB
SHMOO_LIMIT_1MB       = (1 << 20)         #1 MB
SHMOO_LIMIT_16MB      = (1 << 24)         #16 MB
SHMOO_LIMIT_32MB      = (1 << 25)         #32 MB


#enums, project
#enum testMode { QUICK_MODE, RANGE_MODE, SHMOO_MODE };
#enum memcpyKind { DEVICE_TO_HOST, HOST_TO_DEVICE, DEVICE_TO_DEVICE };
#enum printMode { USER_READABLE, CSV };
#enum memoryMode { PINNED, PAGEABLE };

# functions
#void runTest(const int argc, const char **argv);
#void testBandwidth( unsigned int start, unsigned int end, unsigned int increment, testMode mode, memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice);
#void testBandwidthQuick(unsigned int size, memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice);
#void testBandwidthRange(unsigned int start, unsigned int end, unsigned int increment, memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice);
#void testBandwidthShmoo(memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice);
#float testDeviceToHostTransfer(unsigned int memSize, memoryMode memMode);
#float testHostToDeviceTransfer(unsigned int memSize, memoryMode memMode);
#float testDeviceToDeviceTransfer(unsigned int memSize);
#void printResultsReadable(unsigned int *memSizes, float *bandwidths, unsigned int count);
#void printResultsCSV(unsigned int *memSizes, float *bandwidths, unsigned int count);
#void printHelp(void);

def main():
    start = DEFAULT_SIZE
    end = DEFAULT_SIZE
    startDevice = 0
    endDevice = 0
    increment = DEFAULT_INCREMENT
    testMode mode = QUICK_MODE
    htod = False
    dtoh = False
    dtod = False

    modeStr = ''
    device = None
    printMode printmode = USER_READABLE
    char *memModeStr = NULL
    memoryMode memMode = PAGEABLE

    #process command line args
    if(cutCheckCmdLineFlag( argc, argv, "help"))
    {
        printHelp();
        return;
    }

    if(cutCheckCmdLineFlag( argc, argv, "csv"))
    {
        printmode = CSV;
    }

    if( cutGetCmdLineArgumentstr(argc, argv, "memory", &memModeStr) )
    {
        if( strcmp(memModeStr, "pageable") == 0 )
        {
            memMode = PAGEABLE;
        }
        else if( strcmp(memModeStr, "pinned") == 0)
        {
            memMode = PINNED;
        }
        else
        {
            printf("Invalid memory mode - valid modes are pageable or pinned\n");
            printf("See --help for more information\n");
            return;
        }
    }
    else
    {
        #default - pageable memory
        memMode = PAGEABLE;
    }

    if( cutGetCmdLineArgumentstr(argc, argv, "device", &device) )
    {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if( deviceCount == 0 )
        {
            printf("!!!!!No devices found!!!!!\n");
            return;
        } 
        if( strcmp (device, "all") == 0 )
        {
            printf ("\n!!!!!Cumulative Bandwidth to be computed from all the devices !!!!!!\n\n");
            startDevice = 0;
            endDevice = deviceCount-1;
        }
        else
        {
            startDevice = endDevice = atoi(device);
            if( startDevice >= deviceCount || startDevice < 0)
            {
                printf("\n!!!!!Invalid GPU number %d given hence default gpu %d will be used !!!!!\n", startDevice,0);
                startDevice = endDevice = 0;
            }
        }
    }
    printf("Running on......\n");
    for( int currentDevice = startDevice; currentDevice <= endDevice; currentDevice++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, currentDevice);
        printf ("      device %d:%s\n", currentDevice,deviceProp.name);
    }

    if( cutGetCmdLineArgumentstr(argc, argv, "mode", &modeStr) )
    {
        #figure out the mode
        if( strcmp(modeStr, "quick") == 0 )
        {
            mode = QUICK_MODE;
        }
        else if( strcmp(modeStr, "shmoo") == 0 )
        {
            mode = SHMOO_MODE;
        }
        else if( strcmp(modeStr, "range") == 0 )
        {
            mode = RANGE_MODE;
        }
        else
        {
            printf("Invalid mode - valid modes are quick, range, or shmoo\n");
            printf("See --help for more information\n");
            return;
        }
    }
    else
    {
        #default mode - quick
        mode = QUICK_MODE;
    }
    
    if(cutCheckCmdLineFlag( argc, argv, "htod"))
        htod = true;
    if(cutCheckCmdLineFlag( argc, argv, "dtoh"))
        dtoh = true;
    if(cutCheckCmdLineFlag( argc, argv, "dtod"))
        dtod = true;

    if( !htod && !dtoh && !dtod )
    {
        #default:  All
        htod = true;
        dtoh = true;
        dtod = true;
    }

    if( RANGE_MODE == mode )
    {
        if( cutGetCmdLineArgumenti( argc, argv, "start", &start) )
        {
            if( start <= 0 )
            {
                printf("Illegal argument - start must be greater than zero\n");
                return;
            }   
        }
        else
        {
            printf("Must specify a starting size in range mode\n");
            printf("See --help for more information\n");
            return;
        }

        if( cutGetCmdLineArgumenti( argc, argv, "end", &end) )
        {
            if( end <= 0 )
            {
                printf("Illegal argument - end must be greater than zero\n");
                return;
            }

            if( start > end )
            {
                printf("Illegal argument - start is greater than end\n");
                return;
            }
        }
        else
        {
            printf("Must specify an end size in range mode.\n");
            printf("See --help for more information\n");
            return;
        }


        if( cutGetCmdLineArgumenti( argc, argv, "increment", &increment) )
        {
            if( increment <= 0 )
            {
                printf("Illegal argument - increment must be greater than zero\n");
                return;
            }
        }
        else
        {
            printf("Must specify an increment in user mode\n");
            printf("See --help for more information\n");
            return;
        }
    }
   
    if( htod )
    {
        testBandwidth((unsigned int)start, (unsigned int)end, (unsigned int)increment, 
                       mode, HOST_TO_DEVICE, printmode, memMode,startDevice, endDevice);
    }                       
    if( dtoh )
    {
        testBandwidth((unsigned int)start, (unsigned int)end, (unsigned int)increment,
                       mode, DEVICE_TO_HOST, printmode, memMode, startDevice, endDevice);
    }                       
    if( dtod )
    {
        testBandwidth((unsigned int)start, (unsigned int)end, (unsigned int)increment,
                      mode, DEVICE_TO_DEVICE, printmode, memMode, startDevice, endDevice);
    }                       

    printf("&&&& Test PASSED\n");

    cutFree( memModeStr); 


########################################
##  Run a bandwidth test
########################################
def test_bandwidth(start, end, increment, mode, kind, printmode, memMode, startDevice, endDevice):
    """
        unsigned int start
        unsigned int end
        unsigned int increment
        testMode mode
        memcpyKind kind
        printMode printmode
        memoryMode memMode
        int startDevice
        int endDevice
    """

    if mode == QUICK_MODE:
        print "Quick Mode"
        testBandwidthQuick( DEFAULT_SIZE, kind, printmode, memMode, startDevice, endDevice )
    elif mode == RANGE_MODE:
        print "Range Mode"
        testBandwidthRange(start, end, increment, kind, printmode, memMode, startDevice, endDevice)
    elif mode == SHMOO_MODE: 
        print "Shmoo Mode"
        testBandwidthShmoo(kind, printmode, memMode, startDevice, endDevice)
    else:
        print "Invalid testing mode"


####################################
##  Run a quick mode bandwidth test
####################################
def testBandwidthQuick(size, kind, printmode, memMode, startDevice, endDevice):
    """
        unsigned int size
        memcpyKind kind
        printMode printmode
        memoryMode memMode
        int startDevice
        int endDevice
    """
    testBandwidthRange(size, size, DEFAULT_INCREMENT, kind, printmode, memMode, startDevice, endDevice)

####################################/
##  Run a range mode bandwidth test
####################################
def testBandwidthRange(start, end, increment, kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice):
    #count the number of copies we're going to run
    unsigned int count = 1 + ((end - start) / increment);
    
    unsigned int *memSizes = ( unsigned int * )malloc( count * sizeof( unsigned int ) );
    float *bandwidths = ( float * ) malloc( count * sizeof(float) );

    #print information for use
    switch(kind)
    {
    case DEVICE_TO_HOST:    printf("Device to Host Bandwidth for ");
        break;
    case HOST_TO_DEVICE:    printf("Host to Device Bandwidth for ");
        break;
    case DEVICE_TO_DEVICE:  printf("Device to Device Bandwidth\n");
        break;
    }
    if( DEVICE_TO_DEVICE != kind )
    {   switch(memMode)
        {
        case PAGEABLE:  printf("Pageable memory\n");
            break;
        case PINNED:    printf("Pinned memory\n");
            break;
        }
    }

    # Before calculating the cumulative bandwidth, initialize bandwidths array to NULL
    for (int i = 0; i < count; i++)
        bandwidths[i] = 0.0f;

    # Use the device asked by the user
    for (int currentDevice = startDevice; currentDevice <= endDevice; currentDevice++)
    {
        
        cudaSetDevice(currentDevice);
        #run each of the copies
        for(unsigned int i = 0; i < count; i++)
        {
            memSizes[i] = start + i * increment;
            switch(kind)
            {
            case DEVICE_TO_HOST:    bandwidths[i] += testDeviceToHostTransfer( memSizes[i], memMode );
                break;
            case HOST_TO_DEVICE:    bandwidths[i] += testHostToDeviceTransfer( memSizes[i], memMode );
                break;
            case DEVICE_TO_DEVICE:  bandwidths[i] += testDeviceToDeviceTransfer( memSizes[i] );
                break;
            }
            printf(".");
        }
        
        cudaThreadExit();
    } # Complete the bandwidth computation on all the devices

    printf("\n");
    #print results
    if(printmode == CSV)
    {
        printResultsCSV(memSizes, bandwidths, count);
    }
    else
    {
        printResultsReadable(memSizes, bandwidths, count);
    }

    #clean up
    free(memSizes);
    free(bandwidths);

########################################
## Intense shmoo mode - covers a large range of values with varying increments
########################################
def testBandwidthShmoo(memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice):
    #print info for user
    switch(kind)
    {
    case DEVICE_TO_HOST:    printf("Device to Host Bandwidth for ");
        break;
    case HOST_TO_DEVICE:    printf("Host to Device Bandwidth for ");
        break;
    case DEVICE_TO_DEVICE:  printf("Device to Device Bandwidth\n");
        break;
    }
    if( DEVICE_TO_DEVICE != kind )
    {   switch(memMode)
        {
        case PAGEABLE:  printf("Pageable memory\n");
            break;
        case PINNED:    printf("Pinned memory\n");
            break;
        }
    }

    #count the number of copies to make
    unsigned int count = 1 + (SHMOO_LIMIT_20KB  / SHMOO_INCREMENT_1KB)
                        + ((SHMOO_LIMIT_50KB - SHMOO_LIMIT_20KB) / SHMOO_INCREMENT_2KB)
                        + ((SHMOO_LIMIT_100KB - SHMOO_LIMIT_50KB) / SHMOO_INCREMENT_10KB)
                        + ((SHMOO_LIMIT_1MB - SHMOO_LIMIT_100KB) / SHMOO_INCREMENT_100KB)
                        + ((SHMOO_LIMIT_16MB - SHMOO_LIMIT_1MB) / SHMOO_INCREMENT_1MB)
                        + ((SHMOO_LIMIT_32MB - SHMOO_LIMIT_16MB) / SHMOO_INCREMENT_2MB)
                        + ((SHMOO_MEMSIZE_MAX - SHMOO_LIMIT_32MB) / SHMOO_INCREMENT_4MB);

    unsigned int *memSizes = ( unsigned int * )malloc( count * sizeof( unsigned int ) );
    float *bandwidths = ( float * ) malloc( count * sizeof(float) );


    # Before calculating the cumulative bandwidth, initialize bandwidths array to NULL
    for (int i = 0; i < count; i++)
        bandwidths[i] = 0.0f;
   
    # Use the device asked by the user
    for (int currentDevice = startDevice; currentDevice <= endDevice; currentDevice++)
    {
        cudaSetDevice(currentDevice);
        #Run the shmoo
        int iteration = 0;
        unsigned int memSize = 0;
        while( memSize <= SHMOO_MEMSIZE_MAX )
        {
            if( memSize < SHMOO_LIMIT_20KB )
            {
                memSize += SHMOO_INCREMENT_1KB;
            }
            else if( memSize < SHMOO_LIMIT_50KB )
            {
                memSize += SHMOO_INCREMENT_2KB;
            }else if( memSize < SHMOO_LIMIT_100KB )
            {
                memSize += SHMOO_INCREMENT_10KB;
            }else if( memSize < SHMOO_LIMIT_1MB )
            {
                memSize += SHMOO_INCREMENT_100KB;
            }else if( memSize < SHMOO_LIMIT_16MB )
            {
                memSize += SHMOO_INCREMENT_1MB;
            }else if( memSize < SHMOO_LIMIT_32MB )
            {
                memSize += SHMOO_INCREMENT_2MB;
            }else 
            {
                memSize += SHMOO_INCREMENT_4MB;
            }

            memSizes[iteration] = memSize;
            switch(kind)
            {
            case DEVICE_TO_HOST:    bandwidths[iteration] += testDeviceToHostTransfer( memSizes[iteration], memMode );
                break;
            case HOST_TO_DEVICE:    bandwidths[iteration] += testHostToDeviceTransfer( memSizes[iteration], memMode );
                break;
            case DEVICE_TO_DEVICE:  bandwidths[iteration] += testDeviceToDeviceTransfer( memSizes[iteration] );
                break;
            }
            iteration++;
            printf(".");
       }
    } # Complete the bandwidth computation on all the devices

    printf("\n");
    #print results
    if( CSV == printmode)
    {
        printResultsCSV(memSizes, bandwidths, count);
    }
    else
    {
        printResultsReadable(memSizes, bandwidths, count);
    }

    #clean up
    free(memSizes);
    free(bandwidths);

########################################/
##  test the bandwidth of a device to host memcopy of a specific size
########################################/
def testDeviceToHostTransfer(unsigned int memSize, memoryMode memMode):
    unsigned int timer = 0;
    float elapsedTimeInMs = 0.0f;
    float bandwidthInMBs = 0.0f;
    unsigned char *h_idata = NULL;
    unsigned char *h_odata = NULL;

    cutilCheckError( cutCreateTimer( &timer ) );
    
    #allocate host memory
    if( PINNED == memMode )
    {
        #pinned memory mode - use special function to get OS-pinned memory
        cutilSafeCall( cudaMallocHost( (void**)&h_idata, memSize ) );
        cutilSafeCall( cudaMallocHost( (void**)&h_odata, memSize ) );
    }
    else
    {
        #pageable memory mode - use malloc
        h_idata = (unsigned char *)malloc( memSize );
        h_odata = (unsigned char *)malloc( memSize );
    }
    #initialize the memory
    for(unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
    {
        h_idata[i] = (unsigned char) (i & 0xff);
    }

    # allocate device memory
    unsigned char* d_idata;
    cutilSafeCall( cudaMalloc( (void**) &d_idata, memSize));

    #initialize the device memory
    cutilSafeCall( cudaMemcpy( d_idata, h_idata, memSize,
                                cudaMemcpyHostToDevice) );

    #copy data from GPU to Host
    cutilCheckError( cutStartTimer( timer));
    for( unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++ )
    {
        cutilSafeCall( cudaMemcpy( h_odata, d_idata, memSize,
                                cudaMemcpyDeviceToHost) );
    }
   
    #note:  Since Device to Host memcopies are blocking, there is no need
    #       for a cudaThreadSynchronize() here.

    #get the the total elapsed time in ms
    cutilCheckError( cutStopTimer( timer));
    elapsedTimeInMs = cutGetTimerValue( timer);
    
    #calculate bandwidth in MB/s
    bandwidthInMBs = (1e3f * memSize * (float)MEMCOPY_ITERATIONS) / 
                                        (elapsedTimeInMs * (float)(1 << 20));

    #clean up memory
    cutilCheckError( cutDeleteTimer( timer));
    if( PINNED == memMode )
    {
        cutilSafeCall( cudaFreeHost(h_idata) );
        cutilSafeCall( cudaFreeHost(h_odata) );
    }
    else
    {
        free(h_idata);
        free(h_odata);
    }
    cutilSafeCall(cudaFree(d_idata));
    
    return bandwidthInMBs;

########################################/
## test the bandwidth of a host to device memcopy of a specific size
########################################/
def testHostToDeviceTransfer(unsigned int memSize, memoryMode memMode):
    unsigned int timer = 0;
    float elapsedTimeInMs = 0.0f;
    float bandwidthInMBs = 0.0f;
    cutilCheckError( cutCreateTimer( &timer ) );

    #allocate host memory
    unsigned char *h_odata = NULL;
    if( PINNED == memMode )
    {
        #pinned memory mode - use special function to get OS-pinned memory
        cutilSafeCall( cudaMallocHost( (void**)&h_odata, memSize ) );
    }
    else
    {
        #pageable memory mode - use malloc
        h_odata = (unsigned char *)malloc( memSize );
    }
    unsigned char *h_cacheClear1 = (unsigned char *)malloc( CACHE_CLEAR_SIZE );
    unsigned char *h_cacheClear2 = (unsigned char *)malloc( CACHE_CLEAR_SIZE );
    #initialize the memory
    for(unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
    {
        h_odata[i] = (unsigned char) (i & 0xff);
    }
    for(unsigned int i = 0; i < CACHE_CLEAR_SIZE / sizeof(unsigned char); i++)
    {
        h_cacheClear1[i] = (unsigned char) (i & 0xff);
        h_cacheClear2[i] = (unsigned char) (0xff - (i & 0xff));
    }

    #allocate device memory
    unsigned char* d_idata;
    cutilSafeCall( cudaMalloc( (void**) &d_idata, memSize));

    #copy host memory to device memory
    for(unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
    {
        cutilCheckError( cutStartTimer( timer));
        cutilSafeCall( cudaMemcpy( d_idata, h_odata, memSize,
                                cudaMemcpyHostToDevice) );
        
        #Note:  since Host to Device memory copies are blocking,
        #       there is no need for a cudaThreadSynchronize() here.

        #the the total elapsed time in ms
        cutilCheckError( cutStopTimer( timer));
        elapsedTimeInMs += cutGetTimerValue( timer);
        cutilCheckError( cutResetTimer( timer));
        
        #prevent unrealistic caching effects by copying a large amount of data
        for(unsigned int j = 0 ; j < CACHE_CLEAR_SIZE / sizeof(unsigned char); j++)
        {
            h_cacheClear1[i] = h_cacheClear2[i] & i;
        }
    }

    #calculate bandwidth in MB/s
    bandwidthInMBs = (1e3f * memSize * (float)MEMCOPY_ITERATIONS) / 
                                        (elapsedTimeInMs * (float)(1 << 20));

    #clean up memory
    cutilCheckError( cutDeleteTimer( timer));
    if( PINNED == memMode )
    {
        cutilSafeCall( cudaFreeHost(h_odata) );
    }
    else
    {
        free(h_odata);
    }
    free(h_cacheClear1);
    free(h_cacheClear2);
    cutilSafeCall(cudaFree(d_idata));

    return bandwidthInMBs;

########################################/
##! test the bandwidth of a device to device memcopy of a specific size
########################################/
def testDeviceToDeviceTransfer(unsigned int memSize):
    unsigned int timer = 0;
    float elapsedTimeInMs = 0.0f;
    float bandwidthInMBs = 0.0f;
    cutilCheckError( cutCreateTimer( &timer ) );

    #allocate host memory
    unsigned char *h_idata = (unsigned char *)malloc( memSize );
    #initialize the host memory
    for(unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
    {
        h_idata[i] = (unsigned char) (i & 0xff);
    }

    #allocate device memory
    unsigned char *d_idata;
    cutilSafeCall( cudaMalloc( (void**) &d_idata, memSize));
    unsigned char *d_odata;
    cutilSafeCall( cudaMalloc( (void**) &d_odata, memSize));

    #initialize memory
    cutilSafeCall( cudaMemcpy( d_idata, h_idata, memSize,
                                cudaMemcpyHostToDevice) );

    #run the memcopy
    cutilCheckError( cutStartTimer( timer));
    for( unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++ )
    {
        cutilSafeCall( cudaMemcpy( d_odata, d_idata, memSize,
                                cudaMemcpyDeviceToDevice) );
    }
  
    #Since device to device memory copies are non-blocking,
    #cudaThreadSynchronize() is required in order to get
    #proper timing.
    cutilSafeCall( cudaThreadSynchronize() );

    #get the the total elapsed time in ms
    cutilCheckError( cutStopTimer( timer));
    elapsedTimeInMs = cutGetTimerValue( timer);
    
    #calculate bandwidth in MB/s
    bandwidthInMBs = 2.0f * (1e3f * memSize * (float)MEMCOPY_ITERATIONS) / 
                                        (elapsedTimeInMs * (float)(1 << 20));
    
    #clean up memory
    cutilCheckError( cutDeleteTimer( timer));
    free(h_idata);
    cutilSafeCall(cudaFree(d_idata));
    cutilSafeCall(cudaFree(d_odata));

    return bandwidthInMBs;

#############################/
##print results in an easily read format
#############################
def printResultsReadable(unsigned int *memSizes, float *bandwidths, unsigned int count):
    printf("Transfer Size (Bytes)\tBandwidth(MB/s)\n");
    for(unsigned int i = 0; i < count; i++)
    {
        printf("%9u\t\t%.1f\n", memSizes[i], bandwidths[i]);
    }
    printf("\n");
    fflush(stdout);

######################################/
##print results in CSV format
######################################/
def printResultsCSV(unsigned int *memSizes, float *bandwidths, unsigned int count):
    printf("Transfer size (Bytes),");
    for(unsigned int i = 0; i < count; i++)
    {
        printf("%u,", memSizes[i]);
    }
    printf("\n");

    printf("Bandwidth (MB/s),");
    for(unsigned int i = 0; i < count; i++)
    {
        printf("%.1f,", bandwidths[i]);
    }
    printf("\n\n");
    fflush(stdout);

######################################/
##Print help screen
######################################/
def printHelp():
    print    "Usage:  bandwidthTest [OPTION]...\n"
    print    "Test the bandwidth for device to host, host to device, and device to device transfers\n"
    print    "\n"
    print    "Example:  measure the bandwidth of device to host pinned memory copies in the range 1024 Bytes to 102400 Bytes in 1024 Byte increments\n"
    print    "./bandwidthTest --memory=pinned --mode=range --start=1024 --end=102400 --increment=1024 --dtoh\n"

    print    "\n"
    print    "Options:\n"
    print    "--help\tDisplay this help menu\n"
    print    "--csv\tPrint results as a CSV\n"
    print    "--device=[deviceno]\tSpecify the device device to be used\n"
    print    "  all - compute cumulative bandwidth on all the devices\n"
    print    "  0,1,2,...,n - Specify any particular device to be used\n"
    print    "--memory=[MEMMODE]\tSpecify which memory mode to use\n"
    print    "  pageable - pageable memory\n"
    print    "  pinned   - non-pageable system memory\n"
    print    "--mode=[MODE]\tSpecify the mode to use\n"
    print    "  quick - performs a quick measurement\n"
    print    "  range - measures a user-specified range of values\n"
    print    "  shmoo - performs an intense shmoo of a large range of values\n"

    print    "--htod\tMeasure host to device transfers\n"   
    print    "--dtoh\tMeasure device to host transfers\n"
    print    "--dtod\tMeasure device to device transfers\n"
    
    print    "Range mode options\n"
    print    "--start=[SIZE]\tStarting transfer size in bytes\n"
    print    "--end=[SIZE]\tEnding transfer size in bytes\n"
    print    "--increment=[SIZE]\tIncrement size in bytes\n"
