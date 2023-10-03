#include <stdint.h>
typedef uint8_t byte;
struct _Dataset {
    const char *images_file;
    const char *labels_file;
    int image_len;
    int dataset_len;
    double **images;
    int *labels;  
};

typedef struct _Dataset Dataset;

Dataset *load_dataset(const char *images_file, const char *labels_file);
void free_dataset(Dataset *dataset);