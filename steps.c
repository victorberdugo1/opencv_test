#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <pigpio.h>
#include <signal.h>
#include <string.h>
#include <sys/select.h>
#include <sys/time.h>

#define STEP_PIN 27
#define DIR_PIN 17

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

void configurarTerminal()
{
	struct termios tty;
	tcgetattr(STDIN_FILENO, &tty);
	tty.c_lflag &= ~(ICANON | ECHO);
	tcsetattr(STDIN_FILENO, TCSANOW, &tty);
}

void restaurarTerminal()
{
	struct termios tty;
	tcgetattr(STDIN_FILENO, &tty);
	tty.c_lflag |= (ICANON | ECHO);
	tcsetattr(STDIN_FILENO, TCSANOW, &tty);
}

int teclaPresionada()
{
	struct timeval tv = {0, 0};
	fd_set fds;
	FD_ZERO(&fds);
	FD_SET(STDIN_FILENO, &fds);
	return select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0;
}

int leerNumero()
{
	char buffer[10] = {0};
	int index = 0;
	while (index < 9)
	{
		if (teclaPresionada())
		{
			char c;
			read(STDIN_FILENO, &c, 1);
			if (c == '\n') break;
			if (c >= '0' && c <= '9')
			{
				buffer[index++] = c;
				printf("%c", c);
				fflush(stdout);
			}
		}
		usleep(10000);
	}
	return atoi(buffer);
}

int main()
{
	configurarTerminal();

	printf("Usa las flechas para controlar el motor.\n");
	printf("1-5 para cambiar precisión\n");
	printf("R para definir pasos\n");
	printf("S para mover cantidad pasos\n");
	printf("ESC para salir\n");

	int stepSize = 3200 / 20;
	int stepCount = 0;
	int stepRead = 0;

	if (gpioInitialise() < 0) return 1;

	gpioSetMode(STEP_PIN, PI_OUTPUT);
	gpioSetMode(DIR_PIN, PI_OUTPUT);

	// Lanza libcamera-hello en segundo plano
	pid_t cam_pid = fork();
	if (cam_pid < 0)
	{
		fprintf(stderr, "Error al crear el proceso de cámara\n");
		gpioTerminate();
		return 1;
	}
	if (cam_pid == 0)
	{

		int devnull = open("/dev/null", O_WRONLY);
		if (devnull != -1)
		{
			dup2(devnull, STDERR_FILENO);
			close(devnull);
		}


		execlp("libcamera-hello", "libcamera-hello",
				"--qt-preview",
				"--preview", "700,20,640,480",
				"--info-text", "LensPos: %lp",
				"-t", "0",
				(char *)NULL);

		fprintf(stderr, "Error al ejecutar libcamera-hello\n");
		exit(1);
	}


	while (1)
	{
		if (teclaPresionada())
		{
			char buffer[3] = {0};
			read(STDIN_FILENO, buffer, 1);

			if (buffer[0] == 27)
			{
				if (teclaPresionada())
				{
					read(STDIN_FILENO, buffer + 1, 2);
					if (buffer[1] == '[')
					{
						if (buffer[2] == 'C') // flecha derecha
						{
							move_motor(stepSize, 0, 1000);
							stepCount += stepSize;
						}
						else if (buffer[2] == 'D') // flecha izquierda
						{
							move_motor(stepSize, 1, 1000);
							stepCount -= stepSize;
						}
						printf("\r                            \r");
						printf("Pasos acumulados: %d", stepCount);
						fflush(stdout);
					}
				}
				else
				{
					printf("\nSaliendo...\n");
					break;
				}
			}
			else if (buffer[0] >= '1' && buffer[0] <= '5')
			{
				int opciones[5] = {200, 400, 800, 1600, 3200};
				stepSize = opciones[buffer[0] - '1'] / 20;
				printf("\nModo cambiado a %d pasos/rev\n", opciones[buffer[0] - '1']);
			}
			else if (buffer[0] == 'R' || buffer[0] == 'r')
			{
				printf("\nIntroduce el número de pasos actual: ");
				fflush(stdout);
				stepCount = leerNumero();
				printf("\r                                                                            \r");
				printf("Pasos acumulados: %d", stepCount);
				fflush(stdout);
			}
			else if (buffer[0] == 'S' || buffer[0] == 's')
			{
				printf("\nIntroduce el número de pasos a mover: ");
				fflush(stdout);
				stepRead = leerNumero();
				stepCount += stepRead;
				move_motor(stepRead, 0, 1000);
				printf("\r                                                                            \r");
				printf("Pasos acumulados: %d", stepCount);
				fflush(stdout);
			}
		}

		usleep(10000);
	}

	// Mata el proceso de cámara al terminar
	kill(cam_pid, SIGTERM);
	waitpid(cam_pid, NULL, 0);

	restaurarTerminal();
	gpioTerminate();
	return 0;
}
