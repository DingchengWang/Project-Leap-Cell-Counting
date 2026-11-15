/* Exact original-grid block statistics. Project-owned code; system APIs only. */
#include <stdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <dispatch/dispatch.h>
#define API __attribute__((visibility("default")))
#define BLOCK 8

typedef struct {
    const uint8_t *input;
    size_t height,width,stride,grid_height,grid_width,workers;
    int little_endian;
    void *sum,*sum_sq,*maximum;
    uint64_t *histograms;
} Stats;

API int leap_counting_stats_abi_version(void) { return 1; }

static void scan_u8(void *opaque,size_t worker) {
    Stats *s=(Stats *)opaque;
    uint16_t *sum=(uint16_t *)s->sum;
    uint32_t *squares=(uint32_t *)s->sum_sq;
    uint8_t *maximum=(uint8_t *)s->maximum;
    uint64_t *hist=s->histograms+worker*256;
    size_t first=s->grid_height*worker/s->workers;
    size_t stop=s->grid_height*(worker+1)/s->workers;
    for(size_t gy=first;gy<stop;gy++) {
        size_t y0=gy*BLOCK,y1=y0+BLOCK<s->height?y0+BLOCK:s->height;
        for(size_t gx=0;gx<s->grid_width;gx++) {
            size_t x0=gx*BLOCK,x1=x0+BLOCK<s->width?x0+BLOCK:s->width;
            uint32_t total=0,sq=0;uint8_t peak=0;
            for(size_t y=y0;y<y1;y++) {
                const uint8_t *row=s->input+y*s->stride;
                for(size_t x=x0;x<x1;x++) {
                    uint32_t value=row[x];total+=value;sq+=value*value;
                    if(value>peak)peak=(uint8_t)value;
                    hist[value]++;
                }
            }
            size_t index=gy*s->grid_width+gx;
            sum[index]=(uint16_t)total;squares[index]=sq;maximum[index]=peak;
        }
    }
}

static void scan_u16(void *opaque,size_t worker) {
    Stats *s=(Stats *)opaque;
    uint32_t *sum=(uint32_t *)s->sum;
    uint64_t *squares=(uint64_t *)s->sum_sq;
    uint16_t *maximum=(uint16_t *)s->maximum;
    uint64_t *hist=s->histograms+worker*65536;
    size_t first=s->grid_height*worker/s->workers;
    size_t stop=s->grid_height*(worker+1)/s->workers;
    for(size_t gy=first;gy<stop;gy++) {
        size_t y0=gy*BLOCK,y1=y0+BLOCK<s->height?y0+BLOCK:s->height;
        for(size_t gx=0;gx<s->grid_width;gx++) {
            size_t x0=gx*BLOCK,x1=x0+BLOCK<s->width?x0+BLOCK:s->width;
            uint32_t total=0;uint64_t sq=0;uint16_t peak=0;
            for(size_t y=y0;y<y1;y++) {
                const uint8_t *row=s->input+y*s->stride;
                for(size_t x=x0;x<x1;x++) {
                    const uint8_t *p=row+x*2;
                    uint32_t value=s->little_endian?((uint32_t)p[0]|((uint32_t)p[1]<<8)):
                                                       (((uint32_t)p[0]<<8)|(uint32_t)p[1]);
                    total+=value;sq+=(uint64_t)value*value;
                    if(value>peak)peak=(uint16_t)value;
                    hist[value]++;
                }
            }
            size_t index=gy*s->grid_width+gx;
            sum[index]=total;squares[index]=sq;maximum[index]=peak;
        }
    }
}

static int validate(const void *input,size_t height,size_t width,size_t stride,size_t bytes,
                    const void *sum,const void *sum_sq,const void *maximum,const void *hist,size_t workers) {
    if(!input||!sum||!sum_sq||!maximum||!hist||!height||!width||!workers||workers>32)return 1;
    if(height>SIZE_MAX-7||width>SIZE_MAX-7||width>SIZE_MAX/bytes||stride<width*bytes)return 1;
    if(height>SIZE_MAX/stride)return 1;
    return 0;
}

API int leap_counting_stats_u8(const uint8_t *input,size_t height,size_t width,size_t row_stride_bytes,
                             uint16_t *sum,uint32_t *sum_sq,uint8_t *maximum,uint64_t *histograms,size_t workers) {
    if(validate(input,height,width,row_stride_bytes,1,sum,sum_sq,maximum,histograms,workers))return 1;
    Stats s={input,height,width,row_stride_bytes,(height+7)/8,(width+7)/8,workers,1,sum,sum_sq,maximum,histograms};
    memset(histograms,0,workers*256*sizeof(uint64_t));
    dispatch_apply_f(workers,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),&s,scan_u8);
    return 0;
}

API int leap_counting_stats_u16(const uint8_t *input,size_t height,size_t width,size_t row_stride_bytes,int little_endian,
                              uint32_t *sum,uint64_t *sum_sq,uint16_t *maximum,uint64_t *histograms,size_t workers) {
    if(validate(input,height,width,row_stride_bytes,2,sum,sum_sq,maximum,histograms,workers)||
       (little_endian!=0&&little_endian!=1))return 1;
    Stats s={input,height,width,row_stride_bytes,(height+7)/8,(width+7)/8,workers,little_endian,sum,sum_sq,maximum,histograms};
    memset(histograms,0,workers*65536*sizeof(uint64_t));
    dispatch_apply_f(workers,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),&s,scan_u16);
    return 0;
}
