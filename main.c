/*
int main()
{
    if (gpioInitialise() < 0)
        return 1;
    gpioSetMode(STEP_PIN, PI_OUTPUT);
    gpioSetMode(DIR_PIN, PI_OUTPUT);

    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "Error en fork()\n");
        gpioTerminate();
        return 1;
    }
    if (pid == 0) {
        setsid();
        execlp("libcamera-vid", "libcamera-vid", "-t", "0", "--width", "640", "--height", "480", "--framerate", "30", NULL);
        fprintf(stderr, "Error al ejecutar libcamera-vid\n");
        gpioTerminate();
        return 1;
    }

    kill(pid, SIGTERM);
    gpioTerminate();
    return 0;
}*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
//#include <pigpio.h>

#define STEP_PIN 27
#define DIR_PIN 22

void move_motor(int step, int direction, int delay)
{
	//gpioWrite(DIR_PIN, direction);
	for (int i = 0; i < step; i++)
	{
	//	gpioWrite(STEP_PIN, 1);
		usleep(delay);
	//	gpioWrite(STEP_PIN, 0);
		usleep(delay);
	}
}


void configurarTerminal() {
    struct termios tty;
    tcgetattr(STDIN_FILENO, &tty);
    tty.c_lflag &= ~(ICANON | ECHO); // Desactivar modo canónico y eco
    tcsetattr(STDIN_FILENO, TCSANOW, &tty);
}

void restaurarTerminal() {
    struct termios tty;
    tcgetattr(STDIN_FILENO, &tty);
    tty.c_lflag |= (ICANON | ECHO); // Restaurar modo canónico y eco
    tcsetattr(STDIN_FILENO, TCSANOW, &tty);
}

int teclaPresionada() {
    struct timeval tv = {0, 0};
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    return select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0;
}

int leerNumero() {
    char buffer[10] = {0};
    int index = 0;
    while (index < 9) {
        if (teclaPresionada()) {
            char c;
            read(STDIN_FILENO, &c, 1);
            if (c == '\n') break;
            if (c >= '0' && c <= '9') {
                buffer[index++] = c;
                printf("%c", c);
                fflush(stdout);
            }
        }
        usleep(10000);
    }
    return atoi(buffer);
}

int main() {
    configurarTerminal();

	//if (gpioInitialise() < 0) return 1;

	//gpioSetMode(STEP_PIN, PI_OUTPUT);
	//gpioSetMode(DIR_PIN, PI_OUTPUT);

    printf("Usa las flechas para controlar el motor.\n");
    printf("1-5 para cambiar precisión\n");
    printf("R para definir pasos\n");
    printf("S para mover cantidad pasos\n");
    printf("ESC para salir\n");

	int stepSize = 10;
	int stepCount = 0;


	while (1) {
		if (teclaPresionada()) {
			char buffer[3] = {0};
			read(STDIN_FILENO, buffer, 1);

			if (buffer[0] == 27) {
				if (teclaPresionada()) {
					read(STDIN_FILENO, buffer + 1, 2);
                    if (buffer[1] == '[') {
                        if (buffer[2] == 'C')
						{
							//move_motor(10, 0, 100);
							stepCount += stepSize;
						}
                        else if (buffer[2] == 'D')
						{
							//move_motor(10, 1, 100);
							stepCount -= stepSize;
						}
                        printf("\r                            \r");
                        printf("Pasos acumulados: %d", stepCount);
                        fflush(stdout);
                    }
                } else {
                    printf("\nSaliendo...\n");
                    break;
                }
            } else if (buffer[0] >= '1' && buffer[0] <= '5') {
                int opciones[5] = {200, 400, 800, 1600, 3200};
                stepSize = opciones[buffer[0] - '1'] / 20;
                printf("\nModo cambiado a %d pasos/rev\n", opciones[buffer[0] - '1']);
            } else if (buffer[0] == 'R' || buffer[0] == 'r') {
                printf("\nIntroduce el número de pasos actual: ");
                fflush(stdout);
                stepCount = leerNumero();
				printf("\r                                                                            \r");
				printf("Pasos acumulados: %d", stepCount);
				fflush(stdout);

            }
			else if (buffer[0] == 'S' || buffer[0] == 's') {
                printf("\nIntroduce el número de pasos a mover: ");
				fflush(stdout);
				stepCount += leerNumero();
				//move_motor(stepCount, 0, 100);
				printf("\r                                                                            \r");
				printf("Pasos acumulados: %d", stepCount);
				fflush(stdout);
			}

        }
        usleep(10000);
    }
    restaurarTerminal();
	//gpioTerminate();
    return 0;
}

