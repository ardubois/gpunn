#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#define FC 0
#define Relu 1

typedef struct // Cria uma STRUCT para armazenar os dados de uma pessoa
{
  int type;
  float *error;
  float *input;
  float *output;
  int in_size;
  int out_size;
  float *weights;
} FCLayer; 

typedef struct 
{
  int type;
  float *error;
  float *input;
  float *output;
  int in_size;
  int out_size;
} ReluLayer; 

typedef struct 
{
  int type;
  float *error;
  float *input;
  float *output;
  int in_size;
  int out_size;
} Layer; 

typedef struct 
{
  int size;
  int in_size;
  int out_size;
  void** layer;
  float* error;
} NN; 


/////////////// MEMORY ALLOCATION
///////////////////////////
void allocate_batch(NN* net,int batch_size)
{
    int net_size = net -> size;
    net -> error = (float*) malloc(batch_size* net->out_size * sizeof(float));
    for(int i=0;i<net_size;i++)
    {
        Layer *layer = (Layer*) net->layer[i];

        switch(layer->type){
            case FC:
                FCLayer *fc_layer = (FCLayer*) layer;
                fc_layer -> error = (float*) malloc(batch_size* fc_layer -> out_size * sizeof(float));
                fc_layer -> output = (float*) malloc(batch_size* fc_layer -> out_size * sizeof(float));
                break;
            case Relu:
                ReluLayer *relu_layer = (ReluLayer*) layer;
                relu_layer -> error     = (float*) malloc(batch_size* relu_layer -> out_size * sizeof(float));
                relu_layer -> output    = (float*) malloc(batch_size* relu_layer -> out_size * sizeof(float));
                break;

            default:
                printf("Layer not known: %d\n",layer->type);
                assert(1);

        }
    }
}

NN* new_nn(int size)
{

    NN *nn = (NN*) malloc(sizeof(NN));

    nn -> size = size;

    nn -> layer = (void**) malloc(size * sizeof(void**));

    return nn;
}

FCLayer * new_FC_layer(int in_size,int out_size)
{
    FCLayer *fc_layer = (FCLayer*) malloc(sizeof(FCLayer));

    fc_layer -> type = FC;
    fc_layer -> weights =  (float*) malloc(in_size*out_size * sizeof(float));

    for(int i = 0; i< in_size*out_size;i++)
    {
        fc_layer -> weights[i] = 0.2* (1-rand()%3) -0.1;
    }
    
    fc_layer -> in_size = in_size;
    fc_layer -> out_size = out_size;

    return fc_layer;
}

ReluLayer* new_Relu_layer(int size)
{
    
    ReluLayer *relu_layer = (ReluLayer*) malloc(sizeof(ReluLayer));

    relu_layer -> type = Relu;
    relu_layer -> in_size = size;
    relu_layer -> out_size = size;
    return relu_layer;
}


/////////// TO DO: FREE

////////////////////// FUNCOES GPU
//////////////////////////////////////////////

void forward(float *in, float *out, float *weights, int m, int n, int k)
{
    float sum; 
 //  printf("forward, m: %d, n: %d, k: %d\n",m,n,k);
   //   out(kxn) = input(kxm) * weights(mxn)
    for (int row = 0; row < k ; row ++)
    {
        for(int col = 0; col < n; col ++)
        {
          sum = 0;
          for(int i = 0; i < m; i++) 
          {
            //printf("sum += in[%d] * weights[%d] --- i=%d,n=%d,k=%d,row=%d,col=%d\n",row * n + i, i * k + col,i,n,k,row,col );
            sum = sum + in[row * m + i] * weights[i * n + col];
          }
          out[row * n + col] = sum;
         }
    }

}

void mse(float *final_error, float *y_pred,float *y_target, int m,int n)
{
    // final_error = np.mean(np.power(y_target - y_pred, 2))

    // final_error = sum((y_target - y_pred)^2)/m*n
    float error = 0;
    for (int i=0;i<m*n;i++)
    {
        float err = y_target[i]-y_pred[i];
        error = error + err * err;
    }

    *final_error = error/(m*n);

}

int arg_max(float*v, int size){
    float max = v[0];
    int resp = 0;
    for (int i=1;i<size; i++)
    {
        if(v[i]>max)
        {
            max = v[i];
            resp = i;
        }
    }
    return resp;
}

int find_winner(float*v, int size){
    for (int i=0;i<size; i++)
    {
        if(v[i]==1)
        {
            return i;
        }
    }
    assert(0);
    return -1;
}

int count_matches(float *y_pred, float *y_target, int size, int batch_size)
{
   int count = 0;
   float *y_pred_ptr = y_pred;
   float *y_target_ptr = y_target;

   for(int i=0;i<batch_size;i++)
   {
     
     if(find_winner(y_target,size) == arg_max(y_pred,size))
     {
        count ++;
        y_pred_ptr += size;
        y_target_ptr += size;
     }
     return count; 
   }
}
void mse_deriv(float *y_pred,float *y_target,float *error, int m,int n)
{
    // error = 2 * (y_pred - y_true) / y_pred.size

    float size = m*n;
    for (int i=0;i<m*n;i++)
    {
        error[i] = 2*( y_pred[i] - y_target[i])/size;
    }

 
}

void relu(float *in, float *out, int r, int c)
{

    for(int i=0;i<r*c; i++)
    {
        if(in[i] >= 0)
        {
            out[i] = in[i];
        }
        else
        {
            out[i] = 0;
        } 
    }
}

void relu_deriv(float *in, float *out, int r, int c)
{

    for(int i=0;i<r*c; i++)
    {
        if(in[i] > 0)
        {
            out[i] = 1;
        }
        else
        {
            out[i] = 0;
        } 
    }
}

void backward(float *out_error, float *in_error, float *input, float* weights, int m, int n, int b, float lr )
{

    // weights mxn
    // weights error
    // out_error = bxn
    // in_error bxm
    // input bxm



    // in_error (bxm)= out_error(bxn) * transpose(weights) (nxm)
 //printf("backward, m: %d, n: %d, b: %d\n",m,n,b);
    
    for (int row = 0; row < b ; row ++)
    {
        for(int col = 0; col < m; col ++)
        {
          float sum = 0;
          for(int i = 0; i < n; i++) 
          {
            //printf("sum += out_error[%d] * weights[%d] --- i=%d,m=%d,n=%d,b=%d,row=%d,col=%d\n", row * n + i, col * m + i, i, m,n, b, row,col);
            sum = sum + out_error[row * n + i] * weights[col * n + i];
          }
          in_error[row * m + col] = sum;
         }
    }

    // weights(mxn) = weights(mxn) - lr * transpose(input)(mxb) *  out_error(bxn)
       

    for (int row = 0; row < m ; row ++)
    {
        for(int col = 0; col < n; col ++)
        {
          float sum = 0;
          for(int i = 0; i < b; i++) 
          {
            //printf("sum += in[%d] * weights[%d] --- i=%d,n=%d,k=%d,row=%d,col=%d\n",row * n + i, i * k + col,i,n,k,row,col );
            sum = sum + input[i * m + row] * out_error[i * n + col];
          }
          weights[row * n + col] = weights[row * n + col] - lr * sum;
         }
    }

/*
    for (int row = 0; row < m ; row ++)
    {
        for(int col = 0; col < n; col ++)
        {
          float sum = 0;
          for(int i = 0; i < b; i++) 
          {
            sum = sum + input[i * b + row] * out_error[i * n + col];
          }
          weights[row * n + col] = weights[row * n + col] - lr * sum;
         }
    }
*/
}

void predict(NN* nn,float *x_curr,int batch_size){

    
   float *input = x_curr;

   for(int i = 0; i < nn-> size; i++)
   {

     Layer *layer = (Layer*) nn->layer[i];

        switch(layer->type){
            case FC:
              FCLayer* fc_layer = (FCLayer*) layer;
              fc_layer -> input = input;
              
              forward(input, fc_layer -> output,fc_layer-> weights, fc_layer -> in_size, fc_layer -> out_size, batch_size);
              
              input = fc_layer -> output;
              break;
            case Relu:
                ReluLayer *relu_layer = (ReluLayer*) layer;
                relu_layer -> input = input;
              //  printf("before relu size %d\n", relu_layer->in_size);
                relu(input, relu_layer->output, batch_size, relu_layer->in_size);
                //printf("after relu\n");
                input = relu_layer -> output;

             break;

            default:
                printf("Layer not known: %d\n",layer->type);
                assert(0);

        }


   }

}


void update_errors(NN* nn, int batch_size,float lr){

   float *out_error = nn -> error;

   for(int i = nn-> size-1; i >= 0; i--)
   {

     Layer *layer = (Layer*) nn->layer[i];

        switch(layer->type){
            case FC:
              FCLayer* fc_layer = (FCLayer*) layer;
              backward(out_error, fc_layer->error, fc_layer -> input, fc_layer -> weights, 
                        fc_layer -> in_size, fc_layer -> out_size, batch_size, lr);
              out_error = fc_layer -> error;
              break;
            case Relu:
                ReluLayer *relu_layer = (ReluLayer*) layer;
                relu_deriv(out_error, relu_layer-> error, batch_size, relu_layer -> in_size);
                out_error = relu_layer -> error;

             break;

            default:
                printf("Layer not known: %d\n",layer->type);
                assert(1);

        }


   }

 }

void train(NN* nn,  float* x, float* y, int epochs, int batch_size, int train_size, float learning_rate)
{

  

  int x_size = nn -> in_size;
  int y_size = nn -> out_size;

  
  float *x_curr = x;
  float *y_target = y;

  float error = 0;
  int correct_count = 0;

  for(int e = 0; e<epochs;e++)
  {
    for(int i=0;i<train_size;i++)
    {
       
            predict(nn,x_curr,batch_size);
            Layer* last_layer = (Layer*) nn -> layer[nn->size-1];

            float *y_pred = last_layer -> output;

            float curr_error;
        
            mse(&curr_error, y_pred,y_target, batch_size,last_layer-> out_size);
        
            error = error + curr_error;
            //printf("Error: %f\n",error);
        
            correct_count += count_matches(y_pred,y_target, last_layer -> out_size, batch_size);
            //printf("Correct: %d\n",correct_count);
            mse_deriv(y_pred,y_target,nn->error, batch_size,y_size);

            printf("********** nn error ***********\n");
            printf("batch %d y_size %d\n",batch_size,y_size);
            print_matrix(nn->error, batch_size,y_size);
            update_errors(nn,batch_size,learning_rate);

            x_curr += x_size * batch_size;
            y_target += y_size * batch_size;

    }

        printf("Epoch: %d, Error: %f, Correct: %f\n", e, error/train_size, (float)correct_count/train_size);
        x_curr = x;
        y_target = y;
        error = 0;
        correct_count = 0;
  }// epochs

}


void open_file(char *file, int size_entry, int size, float *out)
{
    int line_size = (25*size_entry)+10;
    char line[line_size];

    FILE *fp = fopen(file, "r");
    const char virg[1] = ",";
    char *token;
    char *end_ptr;
    int pos = 0;
    if(fp != NULL)
    {
       
       int lines = 0;
       while(fgets(line, sizeof line, fp) != NULL)
        {   
            lines ++;
            int count = 0;
            
            line[strlen(line)-1]='\0';
            token = strtok(line, virg);
            while (token != NULL) {
                count++;
               // printf("%s\n",token);
                float result = strtof( token, &end_ptr );
                out[pos] = result;
                pos++;
                //printf(":%f\n",result);
                //sscanf(token, "%f", &result);
                //printf("+%f\n",result);
                token = strtok(NULL,virg);
            }
         assert(count == size_entry);
        }
        assert(lines == size);
       fclose(fp);
    } else {
        perror("user.dat");
    }   
}


////// DEBUG/////////////////////////////////////////////////////////////////////
//////////////////////////

void print_matrix(float *matrix,int m,int n)
{  
    int row, column=0;

    for (row=0; row<m; row++)
     {
        for(column=0; column<n; column++)
            {printf("%f     ", matrix[row*n+column]);}
            printf("\n");
     }
}



void init_nn(NN *nn)
{

    for(int i = 0; i< nn->out_size; i++)
    {
        nn-> error[i] = 0;
    }
    for(int i = 0 ; i < nn-> size; i++)
    {
         Layer *layer = (Layer*) nn->layer[i];

        switch(layer->type){
            case FC:
                FCLayer* fc_layer = (FCLayer*) layer;
                for(int j = 0; j < fc_layer ->out_size; j ++)
                {
                    fc_layer -> error[j] =0;
                    fc_layer -> output[j] = 0;
                }
                int value = (fc_layer->out_size * fc_layer -> in_size)/2;
                value = 0 - value;
                for(int j = 0; j < (fc_layer->out_size * fc_layer -> in_size); j ++)
                {
                    fc_layer -> weights[j] = value;
                    value = value +1;
                }
               break;
            case Relu:
                ReluLayer *relu_layer = (ReluLayer*) layer;
                for(int j = 0; j < relu_layer ->out_size; j ++)
                {
                    relu_layer -> error[j] =0;
                    relu_layer -> output[j] = 0;
                }
                break;

            default:
                printf("Layer not known: %d\n",layer->type);
                assert(1);

        }

    }
}

NN* new_nn_debug()
{

 

   NN* nn = new_nn(3);

   nn-> layer[0] = (void*) new_FC_layer(3,4);
   nn-> layer[1] = (void*) new_Relu_layer(4);
   nn-> layer[2] = (void*) new_FC_layer(4,2);

   nn -> in_size = 3;
   nn -> out_size = 2;

   allocate_batch(nn,1);

   init_nn(nn);
    return nn;
}
void print_nn(NN *nn)
{
    printf("=================== NN ==================\n");
    for(int i = 0; i < nn -> size;i++)
    {

        Layer *layer = (Layer*) nn->layer[i];

        switch(layer->type){
            case FC:
              FCLayer* fc_layer = (FCLayer*) layer;
              printf("----- FCLayer[ %d ]--------\n",i);
              printf("in_size: %d\n", fc_layer -> in_size);
              printf("out_size: %d \n",fc_layer -> out_size );
             // printf("input:[%d,%d]\n", 1, fc_layer -> in_size);
             // print_matrix(fc_layer->input,1,fc_layer -> in_size );
              printf("weights[%d,%d]\n", fc_layer -> in_size, fc_layer -> out_size );
              print_matrix(fc_layer-> weights, fc_layer -> in_size, fc_layer -> out_size);
              printf("output:[%d,%d]\n", 1, fc_layer -> out_size);
              print_matrix(fc_layer->output,1,fc_layer -> out_size );
              printf("error:[%d,%d]\n", 1, fc_layer -> out_size);
              print_matrix(fc_layer->error,1, fc_layer -> out_size );
              printf("-------------------------\n");
              break;
            case Relu:
                ReluLayer *relu_layer = (ReluLayer*) layer;
              
              printf("----- Relu [ %d ]--------\n",i);
              printf("in_size: %d\n", relu_layer -> in_size);
              printf("out_size: %d \n",relu_layer -> out_size );
             // printf("input:[%d,%d]\n", 1,relu_layer -> in_size);
             // print_matrix(relu_layer->input,1,relu_layer -> in_size );
              printf("output:[%d,%d]\n", 1,relu_layer -> out_size);
              print_matrix(relu_layer->output,1,relu_layer -> out_size );
              printf("error:[%d,%d]\n", 1, relu_layer -> out_size);
              print_matrix(relu_layer->error,1,relu_layer -> out_size );
              printf("-------------------------\n");

             break;

            default:
                printf("Layer not known: %d\n",layer->type);
                assert(1);

        }


   

    }
}
/*

typedef struct 
{
  int type;
  float *error;
  float *input;
  float *output;
  int in_size;
  int out_size;
  float *weights;
} FCLayer; 

typedef struct 
{
  int size;
  int in_size;
  int out_size;
  void** layer;
  float* error;
} NN; 


typedef struct 
{
  int type;
  float *error;
  float *input;
  float *output;
  int in_size;
  int out_size;
} ReluLayer; 
*/
////////////////////////
void main()
{
   /* 
   int netsize = 5;

   NN* net = new_nn(netsize);

   net-> layer[0] = (void*) new_FC_layer(28*28,512);
   net-> layer[1] = (void*) new_Relu_layer(512);
   net-> layer[2] = (void*) new_FC_layer(512,256);
   net-> layer[3] = (void*) new_Relu_layer(256);
   net-> layer[4] = (void*) new_FC_layer(256,10);

   net -> in_size = 28*28;
   net -> out_size = 10;

   allocate_batch(net,1);




   float x[28*28*1000];
   float y[10*1000];
   open_file("mnist.csv",28*28,1000,x);
   open_file("labels.csv",10,1000,y);
   
   train(net,  x,  y, 5, 1,1000, 0.1);
*/

NN *nn;
printf("%p\n",nn);
nn = new_nn_debug();
printf("%p\n",nn);
print_nn(nn);

                  
    float out[2] = {0,1};
    float in[3] = {1,2,3};
//void train(NN* nn,  float* x, float* y, int epochs, int batch_size, int train_size, float learning_rate)

train(nn, in, out, 1, 1, 1, 0.1);

print_nn(nn);

/*
   float matrix [12] = { 1, 2, 3, 4,
                           5, 6, 7, 8,
                           9, 10, 11, 12
                         }; // 3x4
    float out[4] = {0,0,0,0};
    float in[3] = {1,2,3};
    print_matrix(matrix,3,4);
    
    forward(in, out, matrix, 3, 4, 1);
    printf("Result\n");
    print_matrix(out,1,4);
    float out_error[4] = {1,2,3,4};
    float in_error[3] = {0,0,0};
    backward(out_error, in_error, in, matrix, 3, 4, 1, 0.1 );
    printf("Back\n");
    printf("in_error\n");
    print_matrix(in_error,1,3);
    printf("new_weights\n");
    print_matrix(matrix,3,4);
    float msec;
    print_matrix(out,1,4);
    print_matrix(out_error,1,4);
    mse(&msec, out_error, out, 4, 1);
    printf("mse\n%f\n",msec);
    
    float deriv[4] = {0,0,0,0}; 
   mse_deriv(out_error, out, deriv, 1,4);
   printf("mse deriv\n");
   print_matrix(deriv,1,4);

 */   
}