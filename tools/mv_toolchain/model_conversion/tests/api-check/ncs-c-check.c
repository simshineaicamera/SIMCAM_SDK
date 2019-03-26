#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mvnc.h"

void *loadfile(const char *path, unsigned int *length)
{
	FILE *fp;
	char *buf;

	fp = fopen(path, "rb");
	if(fp == NULL)
		return 0;
	fseek(fp, 0, SEEK_END);
	*length = ftell(fp);
	rewind(fp);
	if(!(buf = malloc(*length)))
	{
		fclose(fp);
		return 0;
	}
	if(fread(buf, 1, *length, fp) != *length)
	{
		fclose(fp);
		free(buf);
		return 0;
	}
	fclose(fp);
	return buf;
}

void runinference(void *graph, void *dev)
{
	char inputtensor[100];
	int i, throttling;
	void *userParam;
	void *result;
	unsigned int resultlen;
	float *timetaken;
	unsigned int timetakenlen, throttlinglen;

	int rc = mvncLoadTensor(graph, inputtensor, sizeof(inputtensor), 0);
	if(rc)
	{
		printf("LoadTensor failed, rc=%d\n", rc);
		return;
	}
	rc = mvncGetResult(graph, &result, &resultlen, &userParam);
	if(rc)
	{
		if(rc == MVNC_MYRIAD_ERROR)
		{
			char *debuginfo;
			unsigned debuginfolen;
			rc = mvncGetGraphOption(graph, MVNC_DEBUG_INFO, (void **)&debuginfo, &debuginfolen);
			if(rc == 0)
			{
				printf("GetResult failed, myriad error: %s\n", debuginfo);
				return;
			}
		}
		printf("GetResult failed, rc=%d\n", rc);
		return;
	}
	printf("Returned %u bytes of result\n", resultlen);
	rc = mvncGetGraphOption(graph, MVNC_TIME_TAKEN, (void **)&timetaken, &timetakenlen);
	if(rc)
	{
		printf("GetGraphOption failed, rc=%d\n", rc);
		return;
	}
	printf("Returned %u bytes of timetaken:\n", timetakenlen);
	timetakenlen = timetakenlen / sizeof(*timetaken);
	float sum = 0;
	for(i = 0; i < timetakenlen; i++)
	{
		printf("%d: %f\n", i, timetaken[i]);
		sum += timetaken[i];
	}
	printf("Total time: %f ms\n", sum);
	rc = mvncGetDeviceOption(dev, MVNC_THERMAL_THROTTLING_LEVEL, (void **)&throttling, &throttlinglen);
	if(rc)
	{
		printf("GetGraphOption failed, rc=%d\n", rc);
		return;
	}
	if(throttling == 1)
		printf("** NCS temperature high - thermal throttling initiated **\n");
	if(throttling == 2)
	{
		printf("*********************** WARNING *************************\n");
		printf("* NCS temperature critical                              *\n");
		printf("* Aggressive thermal throttling initiated               *\n");
		printf("* Continued use may result in device damage             *\n");
		printf("*********************************************************\n");
	}
}

int help()
{
	fprintf(stderr, "./ncs-check [-l<loglevel>] -1  (try one device, open only)\n");
	fprintf(stderr, "./ncs-check [-l<loglevel>] -2  (try two devices, open only)\n");
	fprintf(stderr, "./ncs-check [-l<loglevel>] [-c<count>] <network directory>\n");
	fprintf(stderr, "            <count> is the number of inference iterations, default 2\n");
	fprintf(stderr, "            <network directory> is the directory that contains graph, stat.txt,\n");
	fprintf(stderr, "            categories.txt and inputsize.txt\n");
	fprintf(stderr, "            a dummy picture will be used for inference\n");
	return 0;
}

int main(int argc, char **argv)
{
	char name[MVNC_MAX_NAME_SIZE];
	int rc, i;
	void *h;
	int loglevel = 0, inference_count = 2;
	const char *network = 0;

	for(i = 1; i < argc; i++)
	{
		if(argv[i][0] == '-' && argv[i][1] == 'l')
			loglevel = atoi(argv[i]+2);
		else if(argv[i][0] == '-' && argv[i][1] == 'c')
			inference_count = atoi(argv[i]+2);
		else network = argv[i];
	}
	if(!network)
		return help();
	if(!strcmp(network, "-1"))
		network = 0;
	mvncSetGlobalOption(MVNC_LOG_LEVEL, &loglevel, sizeof(loglevel));
	if(mvncGetDeviceName(0, name, sizeof(name)))
	{
		printf("No devices found\n");
		return -1;
	}
	if( (rc=mvncOpenDevice(name, &h) ))
	{
		printf("OpenDevice %s failed, rc=%d\n", name, rc);
		return -1;
	}

	printf("OpenDevice %s succeeded\n", name);

	if(network)
	{
		if(!strcmp(network, "-2"))
		{
			void *h2;
			if(mvncGetDeviceName(1, name, sizeof(name)))
				fprintf(stderr, "Second device not found\n");
			else {
				if( (rc=mvncOpenDevice(name, &h2) ))
				{
					printf("OpenDevice %s failed, rc=%d\n", name, rc);
					return -1;
				}
				printf("Device closed, rc=%d\n", mvncCloseDevice(h2));
			}
		} else {
			void *graphfile;
			char path[300];
			unsigned len;
			void *g;
			snprintf(path, sizeof(path), "%s/graph", network);
			graphfile = loadfile(path, &len);
			if(graphfile)
			{
				rc = mvncAllocateGraph(h, &g, graphfile, len);
				if(!rc)
				{
					printf("Graph allocated\n");
					for(i = 0; i < inference_count; i++)
						runinference(g, h);
					rc = mvncDeallocateGraph(g);
					printf("Deallocate graph, rc=%d\n", rc);
				} else printf("AllocateGraph failed, rc=%d\n", rc);
				free(graphfile);
			} else fprintf(stderr, "%s/graph not found\n", network);
		}
	}
	printf("Device closed, rc=%d\n", mvncCloseDevice(h));
	return 0;
}

