#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
//#include <pigpio.h>

#define STEP_PIN 27
#define DIR_PIN 17

/*
void move_motor(int step, int direction, int delay)
{
    gpioWrite(DIR_PIN, direction);
	for (int i = 0; i < step; i++)
	{
		gpioWrite(STEP_PIN, 1);
		usleep(delay);
		gpioWrite(STEP_PIN, 0);
		usleep(delay);
	}
}
*/

// Función que ejecuta ffplay para mostrar la preview de la cámara durante 5 segundos.
void tomarPreview()
{
    pid_t pid = fork();
    if (pid < 0)
	{
        perror("fork");
        exit(1);
    }
    if (pid == 0)
	{
        //execlp("ffplay", "ffplay", "-hide_banner", "-loglevel", "quiet", "-t", "5", "/dev/video0", NULL);
        execlp("timeout", "timeout", "2", "ffplay", "-hide_banner", "-loglevel", "quiet", "/dev/video0", NULL);
		perror("execlp");
        exit(1);
    }
	else
	{
        wait(NULL);
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2)
	{
        fprintf(stderr, "Uso: %s <numero_de_pasos>\n", argv[0]);
        return 1;
    }
	/*
	   if (gpioInitialise() < 0) return 1;

	   gpioSetMode(STEP_PIN, PI_OUTPUT);
	   gpioSetMode(DIR_PIN, PI_OUTPUT);
	 */
	int stepSize = atoi(argv[1]);  
	int stepCount = 0;

	printf("Modo: %d pasos por acción\n", stepSize);
    printf("Ejecutando ciclo automático. Presiona Ctrl+C para salir.\n\n");

    while (1) {
        // Ejecuta la preview de la cámara.
        tomarPreview();

        stepCount += stepSize;
        printf("Moviendo motor: %d pasos. Total acumulado: %d\n\n", stepSize, stepCount);
		//move_motor(stepSize, 0, 1000);
        sleep(1);  // Simula tiempo entre iteraciones.
    }
	//gpioTerminate();
    return 0;
}

