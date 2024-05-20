#include <sys/mman.h>
#include <fcntl.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <future>
#include <assert.h>
#include <wordexp.h>
#include <map>
#include <mutex>
#include <iostream>
#include <unistd.h>
#include <string>
#include <cstring>

int main() {

    const std::string SSD_PATH_PREFIX = "~/.cache/esmoe/";
    std::string ssd_path_prefix_;
    

    {
        /* expand tilde */
        wordexp_t exp_result;
        wordexp(SSD_PATH_PREFIX.c_str(), &exp_result, 0);
        ssd_path_prefix_ = std::string(exp_result.we_wordv[0]);
        wordfree(&exp_result);
        std::cerr << "Using ssd path " << ssd_path_prefix_ << std::endl;
    }

    {
        /* check dir exists */
        struct stat dir;
        if (stat(ssd_path_prefix_.c_str(), &dir) != 0) {
            perror("stat");
            exit(0);
        }

        if (!S_ISDIR(dir.st_mode)) {
            std::cerr << "Path " << ssd_path_prefix_ << "does not exist." << std::endl;
            exit(0);
        }
    }

    const size_t size = 16LU * 1024 * 1024 * 1024;
    std::string key("esmoe_shm_test");

    pid_t t = fork();
    bool is_child = t == 0;
    std::string my_alias = is_child ? "child " : "parent ";



    int fd = shm_open(key.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (fd == -1) {
        std::cerr << key << ", Failed to create/open shared memory object: " << std::strerror(errno) << std::endl;
        exit(0);
    }

    if (ftruncate64(fd, size) == -1) {
        std::cout << "ftruncate failed\n" << std::endl;
        exit(0);
    }

    uint8_t *segment_ptr = reinterpret_cast<uint8_t *>(mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (segment_ptr == MAP_FAILED) {
        std::cout << "mmap failed\n" << std::endl;
        exit(0);
    }

    std::cout << "Allocated, filling..." << std::endl;
    memset(segment_ptr, 0, size);
    std::cout << "Filled, waiting..." << std::endl;
    sleep(10);

    if (is_child) {


        // if (ftruncate(fd, 0) == -1) {
        //     perror("ftruncate");
        //     exit(0);
        // }

        std::cout << my_alias << " waiting..." << std::endl;
        sleep(10);


        // if (munmap(segment_ptr, size) == -1) {
        //     perror("munmap");
        //     exit(0);
        // }

        
        // if (shm_unlink(key.c_str()) == -1) {
        //     perror("shm_unlink");
        //     exit(0);
        // }

        // std::cout << my_alias << "Deallocated, waiting..." << std::endl;
        sleep(10);

    } else {
        std::cout << my_alias << "Truncating, waiting..." << std::endl;
        if (ftruncate(fd, 0) == -1) {
            perror("ftruncate");
            exit(0);
        }
        std::cout << my_alias << "Truncated, waiting..." << std::endl;
        sleep(10);

        if (munmap(segment_ptr, size) == -1) {
            perror("munmap");
            exit(0);
        }

        
        if (shm_unlink(key.c_str()) == -1) {
            perror("shm_unlink");
            exit(0);
        }

        std::cout << my_alias << "Deallocated, waiting..." << std::endl;
        sleep(10);
    }



    if  (!is_child) {
        std::cout << "Waiting for child to exit..." << std::endl;
        waitpid(t, NULL, 0);
    }



}