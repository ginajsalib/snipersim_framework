#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// Define the number of threads to be created
#define NUM_THREADS 4

// Function to be executed by each thread
void* perform_work(void* arg)
{
    int index = *((int*)arg);
    printf("Thread %d: starting work...\n", index);
    // Perform some work here
    printf("Thread %d: work done.\n", index);
    // Return NULL to indicate the thread has finished
    // execution
    return NULL;
}

int main()
{
    // Array to hold thread identifiers
    pthread_t threads[NUM_THREADS];
    // Array to hold arguments for each thread
    int thread_args[NUM_THREADS];
    // Variable to hold return code from pthread functions
    int rc;

    for (int i = 0; i < NUM_THREADS; ++i) {
        // Set the thread argument to the current index
        thread_args[i] = i;
        // Create a new thread
        rc = pthread_create(&threads[i], NULL, perform_work,
                            (void*)&thread_args[i]);
        // Check if thread creation was successful
        if (rc) {
            // Print error message if thread creation failed
            printf("Error: unable to create thread, %d\n",
                   rc);
            exit(-1);
        }
    }

    // Loop to join the  threads
    for (int i = 0; i < NUM_THREADS; ++i) {
        // Wait for each thread to complete its execution
        pthread_join(threads[i], NULL);
    }

    return 0;
}
