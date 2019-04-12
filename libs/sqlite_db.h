#include "sqlite3.h"
#include "string.h"
#include "stdio.h"
#include <string>
#include <vector>
using namespace std;

//#define face_feature_lenght 128//mobilefacenet
#define face_feature_lenght 256//lcnn_s
#define MAX_NAME_LENGHT 50
#define MAX_NAME_NUM_IN_DATABASE 500
#define TOP_K 24

//返回值(*指针名)(参数列表)
typedef float (*simility)(float *,float *);

/*根据相似度阈值，在数据库中查找，返回name和simility
参数:p 相似度计算函数指针，feature_query 待查询的特征,ret_name[] 返回的name，ret_simility 和name对应的相似度，
thresh 阈值，ret_len 返回name的长度
*/

int create_db(sqlite3* db);

int  query_db(sqlite3* db, simility ptr_fun, float * feature_query,vector<string> & vec_names,vector<float> & vec_simi,float thresh);
//查询数据库，获取特征值
int  query_get_feature(sqlite3* db, char * name_query,float * ret_feature);
//写入数据库
int insert_db(sqlite3* db, const char * name, float* features);
//删除数据库中的某个字段的数据
int delete_db(sqlite3* db, const char * name);
//更新数据库中的某个字段的数据
int update_db(sqlite3* db, const char * name, float* features);
