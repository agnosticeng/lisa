#ifndef LISA_BRIDGE_H
#define LISA_BRIDGE_H
// UTF-8 JSON request/response. Release every response exactly once.
char *lisa_app_request(const char *request);
void lisa_app_free(char *response);
#endif
